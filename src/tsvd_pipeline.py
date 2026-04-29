from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterable

import jdk4py
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREPARED_DIR = PROJECT_DIR / "prepared"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_SEED = 42
NUMERIC_RAW_COLS = ["u1_norm", "u2_norm", "u3_norm", "i1_norm", "i2_norm", "i3_norm"]
FAULT_TYPES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15]


def configure_java() -> None:
    os.environ.setdefault("JAVA_HOME", str(jdk4py.JAVA_HOME))
    os.environ.setdefault("HADOOP_HOME", str(PROJECT_DIR / "hadoop"))
    os.environ["PATH"] = str(PROJECT_DIR / "hadoop" / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYSPARK_PYTHON", str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe"))
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", os.environ["PYSPARK_PYTHON"])


def spark_session(app_name: str = "TSVD zadanie") -> SparkSession:
    configure_java()
    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.driver.memory", "6g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_single_csv(df: DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.toPandas().to_csv(path, index=False, encoding="utf-8")


def load_measurements(spark: SparkSession) -> DataFrame:
    return (
        spark.read.parquet(str(PREPARED_DIR / "priebehy_spark.parquet"))
        .drop("elektromer")
        .withColumn("t_utc", F.col("t_utc").cast("timestamp"))
    )


def load_faults(spark: SparkSession, measurements: DataFrame) -> DataFrame:
    schema = StructType(
        [
            StructField("eic", StringType(), True),
            StructField("fault_start", StringType(), True),
            StructField("fault_end", StringType(), True),
            StructField("fault_type", IntegerType(), True),
        ]
    )
    raw_faults = (
        spark.read.option("header", True)
        .schema(schema)
        .csv(str(PREPARED_DIR / "poruchy.csv"))
        .where(F.col("eic").isNotNull() & F.col("fault_type").isNotNull())
    )

    bounds = measurements.agg(F.min("t_utc").alias("min_t"), F.max("t_utc").alias("max_t"))
    return (
        raw_faults.crossJoin(bounds)
        .withColumn("start_t", F.coalesce(F.to_timestamp("fault_start"), F.col("min_t")))
        .withColumn("end_date", F.to_date("fault_end"))
        .withColumn(
            "end_t_exclusive",
            F.coalesce(F.to_timestamp(F.date_add("end_date", 1)), F.col("max_t") + F.expr("INTERVAL 1 SECOND")),
        )
        .select("eic", "start_t", "end_t_exclusive", "fault_type")
    )


def integrate_faults(measurements: DataFrame, faults: DataFrame) -> DataFrame:
    measured = measurements.withColumn("measurement_id", F.monotonically_increasing_id())
    joined = measured.join(
        F.broadcast(faults),
        (measured.eic == faults.eic)
        & (measured.t_utc >= faults.start_t)
        & (measured.t_utc < faults.end_t_exclusive),
        "left",
    )

    labels = (
        joined.groupBy("measurement_id")
        .agg(
            F.array_sort(F.collect_set("fault_type")).alias("fault_types_raw"),
            F.max(F.when(F.col("fault_type").isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias("target_binary"),
        )
        .withColumn("fault_types", F.expr("filter(fault_types_raw, x -> x is not null)"))
        .withColumn(
            "target_multiclass",
            F.when(F.size("fault_types") == 0, F.lit("0")).otherwise(F.concat_ws("+", F.col("fault_types"))),
        )
        .drop("fault_types_raw")
    )

    return measured.join(labels, "measurement_id").drop("measurement_id")


def stratified_sample(df: DataFrame, fraction: float = 0.10) -> DataFrame:
    fractions = {row["target_binary"]: fraction for row in df.select("target_binary").distinct().collect()}
    return df.sampleBy("target_binary", fractions=fractions, seed=RANDOM_SEED)


def missing_report(df: DataFrame, columns: Iterable[str]) -> DataFrame:
    exprs = [F.count(F.when(F.col(c).isNull() | F.isnan(c), c)).alias(c) for c in columns]
    return df.agg(*exprs)


def descriptive_stats(df: DataFrame, columns: list[str]) -> DataFrame:
    summary = df.select(columns).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
    skew_kurt = df.agg(
        *[F.skewness(c).alias(f"{c}_skewness") for c in columns],
        *[F.kurtosis(c).alias(f"{c}_kurtosis") for c in columns],
    )
    return summary, skew_kurt


def winsorize_outliers(df: DataFrame, columns: list[str], rel_error: float = 0.01) -> tuple[DataFrame, DataFrame]:
    rows = []
    result = df
    for column in columns:
        q1, q3 = df.approxQuantile(column, [0.25, 0.75], rel_error)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rows.append((column, float(q1), float(q3), float(lower), float(upper)))
        result = result.withColumn(
            column,
            F.when(F.col(column) < lower, lower).when(F.col(column) > upper, upper).otherwise(F.col(column)),
        )
    bounds = df.sparkSession.createDataFrame(rows, ["attribute", "q1", "q3", "lower_bound", "upper_bound"])
    return result, bounds


def impute_missing(df: DataFrame, columns: list[str]) -> DataFrame:
    output_cols = [f"{c}_imputed" for c in columns]
    model = Imputer(strategy="median", inputCols=columns, outputCols=output_cols).fit(df)
    transformed = model.transform(df)
    for original, imputed in zip(columns, output_cols):
        transformed = transformed.drop(original).withColumnRenamed(imputed, original)
    return transformed


def daily_transform(df: DataFrame) -> DataFrame:
    agg_exprs = []
    for c in NUMERIC_RAW_COLS:
        agg_exprs.extend(
            [
                F.avg(c).alias(f"{c}_avg"),
                F.max(c).alias(f"{c}_max"),
                F.min(c).alias(f"{c}_min"),
                F.skewness(c).alias(f"{c}_skewness"),
                F.kurtosis(c).alias(f"{c}_kurtosis"),
            ]
        )

    daily = (
        df.withColumn("day", F.to_date("t_utc"))
        .groupBy("eic", "day")
        .agg(
            F.count("*").alias("n_samples"),
            F.max("target_binary").alias("target_binary"),
            F.array_sort(F.array_distinct(F.flatten(F.collect_list("fault_types")))).alias("daily_fault_types"),
            *agg_exprs,
        )
        .where(F.col("n_samples") >= 144)
        .withColumn(
            "target_multiclass",
            F.when(F.size("daily_fault_types") == 0, F.lit("0")).otherwise(F.concat_ws("+", F.col("daily_fault_types"))),
        )
        .drop("daily_fault_types")
    )

    for fault_type in FAULT_TYPES:
        daily = daily.withColumn(
            f"fault_{fault_type}",
            F.when(F.array_contains(F.split("target_multiclass", r"\+"), str(fault_type)), F.lit(1)).otherwise(F.lit(0)),
        )
    return daily


def train_test_split(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    split_window = Window.partitionBy("target_binary").orderBy(F.rand(RANDOM_SEED))
    counts = df.groupBy("target_binary").count().withColumnRenamed("count", "class_count")
    ranked = (
        df.join(counts, "target_binary")
        .withColumn("rn", F.row_number().over(split_window))
        .withColumn("is_train", F.col("rn") <= F.col("class_count") * F.lit(0.8))
    )
    train = ranked.where("is_train").drop("class_count", "rn", "is_train")
    test = ranked.where(~F.col("is_train")).drop("class_count", "rn", "is_train")
    return train, test


def feature_columns(df: DataFrame) -> list[str]:
    blocked = {
        "eic",
        "day",
        "target_binary",
        "target_multiclass",
        "target_multiclass_index",
        "prediction",
        "rawPrediction",
        "probability",
    }
    return [
        c
        for c, t in df.dtypes
        if c not in blocked and not c.startswith("fault_") and t in {"double", "int", "bigint"}
    ]


def correlation_matrix(train: DataFrame, features: list[str]) -> DataFrame:
    spark = train.sparkSession
    vector_df = VectorAssembler(inputCols=features, outputCol="corr_features", handleInvalid="skip").transform(train)
    matrix = Correlation.corr(vector_df, "corr_features", "pearson").head()[0].toArray().tolist()
    rows = [(features[i], features[j], float(matrix[i][j])) for i in range(len(features)) for j in range(len(features))]
    return spark.createDataFrame(rows, ["attribute_1", "attribute_2", "pearson_correlation"])


def class_weights(df: DataFrame, label_col: str, weight_col: str = "class_weight") -> DataFrame:
    total = df.count()
    counts = df.groupBy(label_col).count()
    n_classes = counts.count()
    weights = counts.withColumn(weight_col, F.lit(total) / (F.lit(n_classes) * F.col("count"))).drop("count")
    return df.join(weights, label_col)


def select_features_by_rf(train: DataFrame, features: list[str], label_col: str, top_n: int = 20) -> tuple[list[str], DataFrame]:
    assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
    rf = RandomForestClassifier(labelCol=label_col, featuresCol="features", numTrees=80, seed=RANDOM_SEED)
    assembled = assembler.transform(train).select(label_col, "features")
    model = rf.fit(assembled)
    importances = [(feature, float(score)) for feature, score in zip(features, model.featureImportances.toArray())]
    importances = sorted(importances, key=lambda x: x[1], reverse=True)
    selected = [feature for feature, score in importances[:top_n] if score > 0] or [feature for feature, _ in importances[:top_n]]
    return selected, train.sparkSession.createDataFrame(importances, ["attribute", "importance"])


def evaluate(predictions: DataFrame, label_col: str, prediction_col: str = "prediction") -> dict[str, float]:
    rows = predictions.groupBy(label_col, prediction_col).count().collect()
    labels = sorted({float(r[label_col]) for r in rows} | {float(r[prediction_col]) for r in rows})
    matrix = {(float(r[label_col]), float(r[prediction_col])): float(r["count"]) for r in rows}
    total = sum(matrix.values())
    if total == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0}

    trace = sum(matrix.get((label, label), 0.0) for label in labels)
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    true_totals = {}
    pred_totals = {}
    for label in labels:
        tp = matrix.get((label, label), 0.0)
        true_total = sum(matrix.get((label, pred), 0.0) for pred in labels)
        pred_total = sum(matrix.get((true, label), 0.0) for true in labels)
        true_totals[label] = true_total
        pred_totals[label] = pred_total
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / true_total if true_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_sum += precision * true_total
        recall_sum += recall * true_total
        f1_sum += f1 * true_total

    numerator = trace * total - sum(pred_totals[label] * true_totals[label] for label in labels)
    denominator_left = total**2 - sum(pred_totals[label] ** 2 for label in labels)
    denominator_right = total**2 - sum(true_totals[label] ** 2 for label in labels)
    denominator = math.sqrt(denominator_left * denominator_right) if denominator_left > 0 and denominator_right > 0 else 0.0
    mcc = numerator / denominator if denominator else 0.0

    return {
        "accuracy": float(trace / total),
        "precision": float(precision_sum / total),
        "recall": float(recall_sum / total),
        "f1": float(f1_sum / total),
        "mcc": float(mcc),
    }


def fit_models(
    train: DataFrame,
    test: DataFrame,
    features: list[str],
    label_col: str,
    multiclass: bool,
) -> tuple[DataFrame, DataFrame]:
    spark = train.sparkSession
    train_weighted = class_weights(train, label_col)

    models = [
        (
            "logistic_regression",
            LogisticRegression(
                featuresCol="features",
                labelCol=label_col,
                weightCol="class_weight",
                family="multinomial" if multiclass else "auto",
                maxIter=60,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                featuresCol="features",
                labelCol=label_col,
                weightCol="class_weight",
                numTrees=120,
                maxDepth=8,
                seed=RANDOM_SEED,
            ),
        ),
    ]
    if multiclass:
        models.append(
            (
                "decision_tree",
                DecisionTreeClassifier(
                    featuresCol="features",
                    labelCol=label_col,
                    weightCol="class_weight",
                    maxDepth=8,
                    seed=RANDOM_SEED,
                ),
            )
        )
    else:
        models.append(
            (
                "gradient_boosted_trees",
                GBTClassifier(
                    featuresCol="features",
                    labelCol=label_col,
                    weightCol="class_weight",
                    maxIter=60,
                    maxDepth=5,
                    seed=RANDOM_SEED,
                ),
            )
        )

    rows = []
    confusion_tables = []
    for model_name, estimator in models:
        pipeline = Pipeline(stages=[VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip"), estimator])
        model = pipeline.fit(train_weighted)
        predictions = model.transform(test)
        metrics = evaluate(predictions, label_col)
        rows.append((model_name, *[metrics[k] for k in ["accuracy", "precision", "recall", "f1", "mcc"]]))
        confusion_tables.append(
            predictions.groupBy(label_col, "prediction")
            .count()
            .withColumn("task", F.lit("multiclass" if multiclass else "binary"))
            .withColumn("model", F.lit(model_name))
        )

    metrics_df = spark.createDataFrame(rows, ["model", "accuracy", "precision", "recall", "f1", "mcc"])
    confusion_df = confusion_tables[0]
    for table in confusion_tables[1:]:
        confusion_df = confusion_df.unionByName(table)
    return metrics_df, confusion_df


def run(sample_fraction: float = 0.10, top_n_features: int = 20) -> dict[str, str]:
    from prepare_inputs import convert_faults, convert_measurements

    convert_measurements()
    convert_faults()
    reset_dir(OUTPUT_DIR)
    spark = spark_session()

    measurements = load_measurements(spark).cache()
    faults = load_faults(spark, measurements).cache()
    integrated = integrate_faults(measurements, faults).cache()
    sampled = stratified_sample(integrated, sample_fraction).cache()
    write_single_csv(sampled.groupBy("target_binary", "target_multiclass").count(), OUTPUT_DIR / "sample_label_distribution.csv")

    write_single_csv(missing_report(integrated, NUMERIC_RAW_COLS), OUTPUT_DIR / "missing_values_raw.csv")
    desc_summary, desc_skew_kurt = descriptive_stats(integrated, NUMERIC_RAW_COLS)
    write_single_csv(desc_summary, OUTPUT_DIR / "descriptive_summary_raw.csv")
    write_single_csv(desc_skew_kurt, OUTPUT_DIR / "descriptive_skew_kurt_raw.csv")

    cleaned, outlier_bounds = winsorize_outliers(integrated, NUMERIC_RAW_COLS)
    cleaned = impute_missing(cleaned, NUMERIC_RAW_COLS).cache()
    write_single_csv(outlier_bounds, OUTPUT_DIR / "outlier_iqr_bounds.csv")

    daily = daily_transform(cleaned).cache()
    train, test = train_test_split(daily)
    train.cache()
    test.cache()
    write_single_csv(daily.groupBy("target_binary", "target_multiclass").count(), OUTPUT_DIR / "daily_label_distribution.csv")

    features = feature_columns(daily)
    write_single_csv(correlation_matrix(train, features), OUTPUT_DIR / "train_correlations.csv")

    selected_binary, binary_importance = select_features_by_rf(train, features, "target_binary", top_n_features)
    write_single_csv(binary_importance, OUTPUT_DIR / "binary_feature_importance.csv")

    indexer = StringIndexer(inputCol="target_multiclass", outputCol="target_multiclass_index", handleInvalid="keep").fit(daily)
    daily_indexed = indexer.transform(daily)
    train_multi, test_multi = train_test_split(daily_indexed)
    multi_features = feature_columns(daily_indexed)
    selected_multi, multi_importance = select_features_by_rf(
        train_multi, multi_features, "target_multiclass_index", top_n_features
    )
    write_single_csv(multi_importance, OUTPUT_DIR / "multiclass_feature_importance.csv")

    binary_metrics, binary_confusion = fit_models(train, test, selected_binary, "target_binary", multiclass=False)
    multi_metrics, multi_confusion = fit_models(
        train_multi, test_multi, selected_multi, "target_multiclass_index", multiclass=True
    )

    write_single_csv(binary_metrics.orderBy(F.desc("f1"), F.desc("mcc")), OUTPUT_DIR / "binary_model_metrics.csv")
    write_single_csv(binary_confusion, OUTPUT_DIR / "binary_confusion_tables.csv")
    write_single_csv(multi_metrics.orderBy(F.desc("f1"), F.desc("mcc")), OUTPUT_DIR / "multiclass_model_metrics.csv")
    write_single_csv(multi_confusion, OUTPUT_DIR / "multiclass_confusion_tables.csv")

    summary = {
        "measurements_rows": measurements.count(),
        "sampled_rows": sampled.count(),
        "daily_rows": daily.count(),
        "train_rows": train.count(),
        "test_rows": test.count(),
        "selected_binary_features": selected_binary,
        "selected_multiclass_features": selected_multi,
        "best_binary_model": binary_metrics.orderBy(F.desc("f1"), F.desc("mcc")).first()["model"],
        "best_multiclass_model": multi_metrics.orderBy(F.desc("f1"), F.desc("mcc")).first()["model"],
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    spark.stop()
    return {k: str(v) for k, v in summary.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-fraction", type=float, default=0.10)
    parser.add_argument("--top-n-features", type=int, default=20)
    args = parser.parse_args()
    summary = run(args.sample_fraction, args.top_n_features)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# TSVD zadanie

Riesenie zadania pre predmet Technologie spracovania velkych dat.

## Struktura

- `dataset/` - povodne vstupy: `priebehy.parquet`, `Poruchy.xlsx`, `popis_dat.docx`
- `prepared/` - Spark-citatelne vstupy vytvorene helperom
- `src/prepare_inputs.py` - technicka priprava vstupov:
  - konverzia Parquet timestampu z nanosekund na mikrosekundy pre Spark 3.5
  - konverzia Excel evidencie poruch na CSV
- `src/tsvd_pipeline.py` - hlavna PySpark pipeline pre integraciu, predspracovanie, transformaciu, trenovanie a vyhodnotenie
- `notebooks/TSVD_zadanie.ipynb` - notebookovy sprievodca riesenim
- `outputs/` - vystupy po spusteni pipeline

## Spustenie

Najprv vloz vstupne subory do priecinka `dataset/`:

- `priebehy.parquet`
- `Poruchy.xlsx`
- `popis_dat.docx`

Potom vytvor virtualne prostredie a nainstaluj zavislosti:

```powershell
cd C:\TSVD_zadanie
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
cd C:\TSVD_zadanie
.\.venv\Scripts\python.exe .\src\tsvd_pipeline.py --sample-fraction 0.10 --top-n-features 20
```

Notebook:

```powershell
cd C:\TSVD_zadanie
.\.venv\Scripts\python.exe -m notebook .\notebooks\TSVD_zadanie.ipynb
```

## Metodicke rozhodnutia

- Poruchy sa integruju intervalovym joinom podla `eic` a casu merania.
- Ak je na jednom odbernom mieste naraz viac poruch, uklada sa mnozina typov poruch.
- Binarny ciel `target_binary` je 1, ak v danom case/dni existuje aspon jedna porucha.
- Multiclass ciel `target_multiclass` je kombinacia sucasnych typov poruch, napriklad `1+4`; bez poruchy je `0`.
- Otvorene intervaly poruch sa doplnaju hranicami dostupnych merani.
- Denné priebehy sa tvoria iba pre cele dni s aspon 144 desatminutovymi meraniami.
- Den s poruchou je oznaceny ako poruchovy, ak sa porucha vyskytla aspon v jednom merani daneho dna.
- Outliery sa spracuju winsorizaciou cez IQR hranice.
- Chybajuce numericke hodnoty sa nahradia medianom.
- Vyber atributov je robeny cez dolezitost atributov z Random Forest modelu.

## Aktualne overene vysledky

Finalny beh s `--sample-fraction 0.10 --top-n-features 20` prebehol uspesne:

- pocet merani: 7 659 371
- pocet riadkov v stratifikovanej vzorke: 764 928
- pocet denných priebehov: 52 449
- train/test: 41 958 / 10 491
- najlepsi binarny model: `gradient_boosted_trees`, F1 = 0.9772, MCC = 0.9549
- najlepsi multiclass model: `random_forest`, F1 = 0.9548, MCC = 0.9335

Detailne metriky su v:

- `outputs/binary_model_metrics.csv`
- `outputs/multiclass_model_metrics.csv`
- `outputs/binary_confusion_tables.csv`
- `outputs/multiclass_confusion_tables.csv`

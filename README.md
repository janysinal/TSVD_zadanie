# TSVD zadanie

Zjednodusene riesenie zadania z predmetu **Technologie spracovania velkych dat**.

Cele riesenie je v jednom notebooku:

- `TSVD_zadanie.ipynb`

Notebook obsahuje:

- nacitanie dat,
- integraciu poruch s priebehmi prudov a napati,
- 10 % stratifikovany sampling,
- pracu s chybajucimi hodnotami a outliermi,
- popisne charakteristiky,
- transformaciu merani na denne priebehy,
- train/test rozdelenie,
- korelacie a vyber atributov,
- trenovanie 3 binarnych klasifikatorov,
- trenovanie 3 multiclass klasifikatorov,
- kontingencne tabulky, precision, recall, F1 a MCC.

## Data

Vstupne subory nie su v repozitari. Pred spustenim ich vloz do priecinka:

```text
C:\TSVD_zadanie\dataset
```

Ocakavane subory:

- `priebehy.parquet`
- `Poruchy.xlsx`
- `popis_dat.docx`

## Instalacia

```powershell
cd C:\TSVD_zadanie
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Spustenie notebooku

```powershell
cd C:\TSVD_zadanie
.\.venv\Scripts\python.exe -m notebook TSVD_zadanie.ipynb
```

## Poznamka

Povodny Parquet subor ma cas ulozeny ako nanosekundovy timestamp. Spark 3.5 ho nevie na Windowse priamo nacitat, preto notebook na zaciatku vytvori lokalnu kopiu:

```text
prepared/priebehy_spark.parquet
```

Tento priecinok je generovany automaticky a nie je sucastou GitHub repozitara.

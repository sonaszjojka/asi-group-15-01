# ASI 15c-01

## Zbiór danych

- Link: <https://archive.ics.uci.edu/dataset/2/adult>
- Data pobrania: 03.10.2025r 16:53

### Licencja zbioru danych

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
This allows for the sharing and adaptation of the datasets for any purpose, provided that the appropriate credit is given.

## Metryka oceny modelu

Jako nasza metryka oceny modelu wybraliśmy **AP** (Average Precision). Decyzja ta jest zmotywowana faktem, że metryka ta znakomicie radzi sobie przy niezbalansowanych zbiorach danych. Dzięki niej będziemy też mogli skupić się na ocenie jakości modelu w przewidywaniu klasy pozytywnej, czyli tego czy dana osoba zarabia >$50k.

Rozważaliśmy również metrykę AUC-ROC, aczkolwiek jak ustaliliśmy, jej wynik może być łatwo zawyżany przez przewagę w ilości wierszy z klasą negatywną (<$50k).

## Wandb

Panel projektu w W&B dostępny jest pod adresem:
<https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01?nw=nwusers28044>

## Kedro Quickstart

Cały pipeline (ładowanie danych, preprocessing, podział na zbiory, trening modelu, ewaluacja, logowanie metryk i modelu) uruchamia się poleceniem:

kedro run

Po wykonaniu pipeline dane i model zapisują się lokalnie w strukturze:

- dane surowe: data/01_raw/
- dane po czyszczeniu: data/02_interim/
- dane przygotowane do uczenia modelu: data/03_processed/
- wytrenowany model: data/06_models/ag_production
- metryki: data/09_tracking/ag_metrics.json

## Kedro Pipeline

![Pipline Screens](images/kedro-pipeline.png)

## Kryterium wyboru modelu

Przy wyborze najlepszego modelu będziemy kierować się następującymi kryteriami:

1. **Główna metryka (Average Precision):** Model musi osiągać jak najwyższą wartość metryki AP na zbiorze testowym.
2. **Czas predykcji:** Średni czas potrzebny na wygenerowanie predykcji dla całego zbioru testowego. Niższa wartość oznacza lepszą wydajność.

## Porównanie modeli

Poniżej znajduje się porównanie przeprowadzonych eksperymentów i ich rezultatów. Więcej danych można znaleźć na dashboardzie [Weights & Biases](https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/workspace).

Warto zaznaczyć, że większość najlepszych modeli wybieranych przez AutoGluona to **WeightedEnsemble**. Są to modele, które łączą predykcje wielu bazowych modeli (np. LightGBM, CatBoost, RandomForest) trenowanych przez AutoGluon, dzięki czemu często osiąga lepsze wyniki niż pojedyncze modele.

### Tabela wyników

| Model                        | Average Precision     | Czas predykcji (s)      | Cel Eksperymentu                         | Link do run'a W&B                                                                                                             |
| ---------------------------- | --------------------- | ----------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| WeightedEnsemble_L2_FULL     | 0.8315796944143125    | 0.16523751669446937     | Baseline                                 | [bumbling-night-114](https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/runs/120oguld) |
| **WeightedEnsemble_L3_FULL** | **0.833053972604723** | **0.10139071250450797** | **Tylko modele Gradient Boosting**       | [**noble-tree-115**](https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/runs/0jru3flo) |
| WeightedEnsemble_L3_FULL     | 0.790352868646671     | 0.17501385431387462     | Tylko modele oparte o sieci neuronowe    | [drawn-darkness-116](https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/runs/ms7i33wf) |
| WeightedEnsemble_L2_FULL     | 0.8326541318393061    | 0.14913365819957108     | Model najwyższej jakości                 | [zesty-shape-117](https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/runs/okytxzhv)    |
| XGBoost_BAG_L1_FULL          | 0.8315290504997589    | **0.03168549991678447** | Model najwyższej jakości bez ensemblingu | [fine-fog-118](https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/runs/nps8zf2q)       |

### Wykres AP

![Wykres Average Precision](images/w&b_ap_chart.svg)

### Wykres czasu predykcji

![Wykres czasu predykcji](images/w&b_avg_prediction_time_chart.svg)

### Wybór najlepszego modelu

W związku z powyższymi wynikami, jako nasz finalny model wybieramy ten wytrenowany podczasu run'a `noble-tree-115`. Jego wybór potwierdza nasze wcześniejsze przypuszczenia, mówiące że najlepiej dla wybranego przez nas zbioru danych sprawdzą się modele bazujące na drzewach decyzyjnych. Zyskał on nie dość, że najlepszy wynik AP spośród wszystkich modeli, ale jednocześnie drugi najlepszy jeśli chodzi o czas predykcji. Wyprzedził go tutaj jedynie model niekorzystający z ensemblingu.

## Wersjonowanie modelu i danych

Najlepsza wersja modelu została zapisana jako artefakt w [Weights & Biases](https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/artifacts/model/ag_model/v46) oraz w lokalnym katalogu w `data/06_models/ag_production`.

Wersja danych, na których model był trenowany, również jest zapisana jako [artefakt W&B](https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/artifacts/dataset/clean_data/v0), co zapewnia pełną odtwarzalność wyników.

## Uruchom

uvicorn src.api.main:app --reload --port 8000

## Test health

curl <http://127.0.0.1:8000/healthz>

## Predykcja Power Shell

```bash
curl.exe -X POST http://127.0.0.1:8000/predict `
 -H "Content-Type: application/json" `
 -d '{\"age\": 67, \"workclass\": \"State-gov\", \"fnlwgt\": 77516, \"education\": \"Bachelors\", \"education-num\": 13, \"marital-status\": \"Never-married\", \"occupation\": \"Adm-clerical\", \"relationship\": \"Not-in-family\", \"race\": \"White\", \"sex\": \"Male\", \"capital-gain\": 2174, \"capital-loss\": 0, \"hours-per-week\": 40, \"native-country\": \"United-States\"}'
```

## Predykcja Linux

```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{ "age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors", "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States" }'
```

## Docker quickstart

```bash
docker compose --build
```

### UI

```bash
streamlit run src/ui/app.py
```

### DB

```bash
 docker exec -it asi-group-15-01-db-1 psql -U app_user -d app_db -c "select * from predictions limit 100;"
```

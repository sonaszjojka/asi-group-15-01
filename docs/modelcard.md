# Model Card

### Problem & Use-case

**Kontekst:**
Średniej wielkości firma analityczna HR, **WorkForceSight**, opracowuje dashboard wspierający podejmowanie decyzji dla badaczy rynku pracy i analityków polityki społecznej. Celem jest dostarczenie wglądu w trendy dystrybucji dochodów w różnych grupach demograficznych, co pomoże organizacjom lepiej zrozumieć dynamikę siły roboczej.

**Problem biznesowy:**
Klienci WorkForceSight często proszą o narzędzia eksploracyjne, które pozwolą im analizować, w jaki sposób konkretne atrybuty demograficzne lub zawodowe korelują z wyższymi progami dochodowymi. Jednak wielu klientów nie posiada wewnętrznych możliwości budowania modeli predykcyjnych. Aby ich wesprzeć, WorkForceSight chce zaoferować prosty, przejrzysty komponent przewidywania dochodów w ramach swojej platformy analitycznej.

**Zamierzone zastosowanie (Intended Use):**
Model został zaprojektowany, aby:
- Identyfikować grupy osób, które z większym prawdopodobieństwem zarabiają powyżej 50 tys. USD rocznie.
- Umożliwić badaczom przeprowadzanie analiz typu „co-jeśli” (np. jak wykształcenie lub liczba przepracowanych godzin wpływają na przewidywany dochód).
- Służyć jako przykład edukacyjny (teaching case) w zakresie interpretowalności ML i analizy sprawiedliwości (fairness).
- Wspierać wewnętrzne symulacje — **nie** służy do podejmowania zautomatyzowanych decyzji.

**Użytkownicy docelowi:**
- Ekonomiści rynku pracy
- Badacze HR
- Analitycy polityki społecznej
- Studenci uczący się koncepcji ML fairness

**Nieprzeznaczone do:**
Podejmowania decyzji o zatrudnieniu, oceny zdolności kredytowej (scoringu) ani żadnych zautomatyzowanych ocen wpływających na życie jednostek.
Przewidywania są przybliżone, poglądowe i nigdy nie powinny być wykorzystywane do rzeczywistych decyzji.

---

### Data (źródło, licencja, rozmiar, PII=brak)

- **Źródło:** <https://archive.ics.uci.edu/dataset/2/adult>
- **Licencja:** CC BY 4.0 — pozwala na udostępnianie i adaptację z uznaniem autorstwa.
- **Rozmiar:** 48 842 rekordy, 14 atrybutów.
- **PII (Dane Osobowe):** Brak informacji umożliwiających identyfikację osób (zbiór kategoryczny i zanonimizowany).

---

### Metrics (główne + pomocnicze, zbiór testowy)

- **Główna metryka:** Average Precision (AP)
- **Metryki pomocnicze:** Accuracy (Dokładność), Precision (Precyzja), Recall (Czułość), F1-Score, AUC-ROC
- **Ewaluacja:** Przeprowadzona na zbiorze testowym wygenerowanym przez pipeline Kedro.
  *(Tutaj wstaw rzeczywiste wyniki, gdy będą dostępne)*

---

### Limitations & Risks (Ograniczenia i Ryzyka)

- **Przestarzały zbiór danych:**
  Zbiór „Adult” odzwierciedla warunki na rynku pracy w USA w latach 90. Przewidywania mogą nie przekładać się na współczesne stanowiska, wynagrodzenia czy warunki ekonomiczne.

- **Ograniczony zakres cech:**
  Na dochód wpływa wiele czynników niereprezentowanych w tym zbiorze (np. trendy branżowe, region geograficzny, poziom doświadczenia). Ogranicza to dokładność predykcji.

- **Potencjalne obciążenia (bias) w kolumnach wrażliwych:**
  Zbiór danych zawiera atrybuty takie jak płeć, rasa i stan cywilny, które mogą wprowadzać statystyczne obciążenia lub odzwierciedlać historyczne nierówności zakorzenione w danych.

- **Niezbalansowanie klas:**
  Stosunek próbek z dochodem >50k do ≤50k jest bardzo nierówny, co może prowadzić do zawyżonej wydajności dla klasy większościowej i słabszych predykcji dla klasy mniejszościowej.

- **Brak interpretacji przyczynowo-skutkowej:**
  Model wyłapuje korelacje, a nie związki przyczynowe. Użytkownicy mogą błędnie interpretować cechy takie jak wykształcenie czy rasa jako mające deterministyczny wpływ na wynik.

- **Przypadki brzegowe:**
  Osoby o nietypowych kombinacjach atrybutów (np. bardzo wysoki wiek przy zerowej liczbie godzin pracy) mogą generować niewiarygodne predykcje ze względu na słabą reprezentację w danych.

---

### Ethics & Risk (Etyka i Ryzyko)

- **Kwestie sprawiedliwości (Fairness):**
  Atrybuty wrażliwe (płeć, rasa, stan cywilny) mogą zawierać istniejące nierówności społeczne. Model może odzwierciedlać lub wzmacniać historyczne uprzedzenia obecne w zbiorze danych.

- **Ryzyko niewłaściwego użycia:**
  Wykorzystanie tego modelu do decyzji o zatrudnieniu, wynagrodzeniu lub kredytach mogłoby skutkować dyskryminującymi wynikami. Nie może on być wdrażany w żadnym kontekście wpływającym na rzeczywiste osoby.

- **Przejrzystość i wyjaśnialność:**
  Użytkownicy mogą błędnie interpretować predykcje modelu jako nakazowe, a nie opisowe. Jasna komunikacja jest kluczowa, aby uniknąć mylących wniosków.

- **Strategie mitygacji (zapobiegania):**
    - Ograniczenie zastosowania modelu wyłącznie do celów edukacyjnych i eksploracyjnych.
    - Zapewnienie dokumentacji podkreślającej zakaz użycia produkcyjnego i decyzyjnego.
    - Unikanie eksponowania cech wrażliwych w zautomatyzowanych procesach, chyba że jest to wymagane do analizy fairness.
    - Wymaganie od użytkowników potwierdzenia, że model nie jest przeznaczony do rzeczywistych wdrożeń.

---

### Versioning

- **W&B Run:** https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/runs/0jru3flo?nw=nwusers27523
- **Model Artifact:** https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/artifacts/model/ag_model/v46/overview
- **Code:** d41c2a29bededf384cb63ca2d1c02502ae953820
- **Data:** `data/02_interim/clean.parquet` lub https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/artifacts/dataset/clean_data/v0
- **Environment:** Python 3.10, AutoGluon 1.1.0, Kedro, scikit-learn, W&B
- **Link do dashboardu z porównaniem:** https://wandb.ai/s28044-polish-japanese-academy-of-information-technology/asi-group-15-01/workspace

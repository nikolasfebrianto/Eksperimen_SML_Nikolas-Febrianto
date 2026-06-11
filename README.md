# Eksperimen_SML_Nikolas_Febrianto

## Dataset

Dataset yang digunakan adalah **Breast Cancer Wisconsin (Diagnostic)** dari **UCI Machine Learning Repository**.

- Sumber: UCI Machine Learning Repository
- URL: `https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic`
- File raw lokal: `breast_cancer_raw/breast_cancer_raw.csv`
- Target asli: `diagnosis` (`M` = malignant, `B` = benign)
- Target modelling: `is_malignant` (`1` = malignant, `0` = benign)

## Struktur

```text
Eksperimen_SML_Nikolas_Febrianto
├── .github/workflows/preprocess.yml
├── .workflow/README.md
├── breast_cancer_raw/breast_cancer_raw.csv
├── preprocessing
│   ├── Eksperimen_Nikolas_Febrianto.ipynb
│   ├── automate_Nikolas_Febrianto.py
│   └── breast_cancer_preprocessing/
├── requirements.txt
└── README.md
```

## Menjalankan preprocessing otomatis

```bash
pip install -r requirements.txt
python preprocessing/automate_Nikolas_Febrianto.py \
  --raw-path breast_cancer_raw/breast_cancer_raw.csv \
  --output-dir preprocessing/breast_cancer_preprocessing
```

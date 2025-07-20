
# Deep Learning Embeddings for Zero-Day Cyber Threat Detection

This project implements a modular and real-time anomaly detection framework inspired by the IEEE paper:  
**"Deep Learning Embeddings for Zero-Day Cyber Threat Detection: A Multi-Modal Framework for Network Flow and SIEM Data Analysis"**

---

## 📁 Project Structure

```
embedding_anomaly_detection/
│
├── data/
│   ├── cse_cic_ds2018.csv                # Raw network flow data (user-provided)
│   ├── synthetic_siem_logs.csv           # Synthetic SIEM event logs
│   └── preprocessing/
│       ├── network_flow_preprocessing.py # Network flow preprocessing
│       └── synthetic_siem_generator.py   # SIEM data generation script
│
├── models/
│   ├── vae/
│   │   ├── model.py                      # Variational Autoencoder
│   │   └── train_vae.py                  # VAE training
│   └── graphsage/
│       ├── model.py                      # GraphSAGE GNN
│       └── train_graphsage.py           # Graph training
│
├── anomaly_detection/
│   └── unsupervised_methods.py          # Isolation Forest, One-Class SVM, LOF
│
├── evaluation/
│   └── metrics.py                        # Evaluation metrics (AUC, F1, etc.)
│
├── experiments/
│   └── run_experiment.py                # Master runner to execute end-to-end
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Install Python packages**:

```bash
pip install -r requirements.txt
```

2. **Download and place network flow data** (CSE-CIC-DS2018):

- Download from: https://www.unb.ca/cic/datasets/cse-cic-ids2018.html
- Use CICFlowMeter to extract flows
- Save as `data/cse_cic_ds2018.csv`

3. **Generate synthetic SIEM data**:

```bash
python data/preprocessing/synthetic_siem_generator.py
```

---

## 🚀 Running the Experiment

```bash
python experiments/run_experiment.py
```

This runs:
- VAE training on network flow
- GraphSAGE training on SIEM logs
- Unsupervised anomaly detection
- Evaluation and reporting of detection metrics

---

## 📊 Sample Output

```
--- Network Flow Detection Results ---
AUC-ROC: 0.9021
F1-score: 0.8460
FPR: 0.0243

--- SIEM Graph Detection Results ---
AUC-ROC: 0.9287
F1-score: 0.8746
FPR: 0.0211
```

---

## 🔬 Methods Used

- **Embeddings**:
  - Variational Autoencoder (VAE) for flow compression
  - GraphSAGE for SIEM relationship modeling

- **Anomaly Detection**:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM

- **Evaluation**:
  - AUC-ROC, AUC-PR
  - F1, Precision, Recall
  - False Positive Rate (FPR)

---

## 📌 Notes

- GraphSAGE is implemented with `torch-geometric`
- All models support real-time processing using mini-batches
- You can replace synthetic data with real SIEM logs

---

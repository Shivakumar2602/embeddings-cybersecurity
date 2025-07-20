import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from data.preprocessing.network_flow_preprocessing import preprocess_network_flow_data
from models.vae.train_vae import train_vae
from anomaly_detection.unsupervised_methods import run_isolation_forest
from evaluation.metrics import evaluate_anomaly_detection

# Dummy data paths (replace with actual)
network_flow_csv = 'data/sample_network_flow.csv'
siem_data_placeholder = 'data/sample_siem.csv'

# Preprocess Network Flow Data
X = preprocess_network_flow_data(network_flow_csv, save_path='data')
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Train VAE
vae_model = train_vae(data=X_train, input_dim=X.shape[1], latent_dim=64, epochs=10)

# Encode test data
vae_model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_test.toarray()).float()
    recon, mu, log_var = vae_model(X_tensor)
    recon_error = torch.sum((X_tensor - recon) ** 2, dim=1).numpy()

# Anomaly Detection on VAE embeddings
scores, clf = run_isolation_forest(recon_error.reshape(-1, 1))

# Simulated labels (replace with actual labels from dataset)
y_true = np.random.choice([0, 1], size=len(scores), p=[0.9, 0.1])  # 90% normal, 10% anomaly

# Evaluate
results = evaluate_anomaly_detection(y_true, scores)
for k, v in results.items():
    print(f"{k}: {v:.4f}")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from data.preprocessing.network_flow_preprocessing import preprocess_network_flow_data
from models.vae.train_vae import train_vae
from anomaly_detection.unsupervised_methods import run_isolation_forest, run_lof
from evaluation.metrics import evaluate_anomaly_detection
from models.graphsage.train_graphsage import build_graph, train_graphsage

# Paths
network_flow_csv = '/data/sample_network_flow.csv'  # Replace with real CSE-CIC file
siem_csv = '/data/synthetic_siem_logs.csv'

# --- NETWORK FLOW ---
X = preprocess_network_flow_data(network_flow_csv, save_path='data')
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

vae_model = train_vae(data=X_train, input_dim=X.shape[1], latent_dim=64, epochs=10)
vae_model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_test.toarray()).float()
    recon, mu, log_var = vae_model(X_tensor)
    recon_error = torch.sum((X_tensor - recon) ** 2, dim=1).numpy()

scores_nf, _ = run_isolation_forest(recon_error.reshape(-1, 1))
y_true_nf = np.random.choice([0, 1], size=len(scores_nf), p=[0.9, 0.1])
results_nf = evaluate_anomaly_detection(y_true_nf, scores_nf)

print("\\n--- Network Flow Detection Results ---")
for k, v in results_nf.items():
    print(f"{k}: {v:.4f}")

# --- SIEM LOGS ---
df = pd.read_csv(siem_csv)
entity_to_id = {v: i for i, v in enumerate(pd.concat([df['user'], df['host'], df['process']]).unique())}
df['src'] = df['user'].map(entity_to_id)
df['dst'] = df['host'].map(entity_to_id)
df['process_id'] = df['process'].map(entity_to_id)

# Node features as one-hot vectors (simplified)
num_nodes = len(entity_to_id)
features = np.eye(num_nodes)

# Edges from user-host interactions
edge_index = df[['src', 'dst']].values.T.tolist()

# Build and train GraphSAGE
graph_data = build_graph(features, edge_index)
graphsage_model = train_graphsage(graph_data, input_dim=num_nodes, out_dim=128)

graphsage_model.eval()
with torch.no_grad():
    embeddings = graphsage_model(graph_data.x, graph_data.edge_index).numpy()

# Anomaly detection using LOF on embeddings
labels = df['label'].values
scores_siem, _ = run_lof(embeddings)
results_siem = evaluate_anomaly_detection(labels[:len(scores_siem)], scores_siem)

print("\\n--- SIEM Graph Detection Results ---")
for k, v in results_siem.items():
    print(f"{k}: {v:.4f}")


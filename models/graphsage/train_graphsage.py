import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models.graphsage.model import GraphSAGE

def build_graph(node_features, edge_index):
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def train_graphsage(graph_data, input_dim, hidden_dim=64, out_dim=128, epochs=20):
    model = GraphSAGE(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=out_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index)
        loss = -torch.mean(out)  # dummy unsupervised loss for example
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model

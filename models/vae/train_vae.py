import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from models.vae.model import VAE

def loss_function(recon_x, x, mu, log_var):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def train_vae(data, input_dim, latent_dim, epochs=50, batch_size=128):
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(data).float()), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, mu, log_var = model(x)
            loss = loss_function(recon, x, mu, log_var)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss}")

    return model

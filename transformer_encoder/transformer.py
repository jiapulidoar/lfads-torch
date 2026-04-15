import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from tqdm import tqdm
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset

class EncTransformer(nn.Module):

    #def __init__(self, n_neurons, encoder_in_dim: int, encoder_out_dim: int, n_heads, n_layers: int) -> None:
    def __init__(self, n_neurons, d_model, n_heads, n_layers, drop_prob=0.2) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
            nn.LayerNorm(n_neurons)
        )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_neurons, nhead=n_heads, batch_first=True, dropout=drop_prob)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        self.to_latent = nn.Sequential(
            nn.Linear(n_neurons, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.from_latent = nn.Sequential(
            nn.Linear(d_model, n_neurons),
            nn.LayerNorm(n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons)
        )

        self.output_proj = nn.Sequential(
            nn.Softplus()
        )

    def positional_encoding(self, X: torch.Tensor) -> torch.Tensor:
        """
        X has shape (batch_size, sequence_length, embedding_dim)

        This function should create the positional encoding matrix
        and return the sum of X and the encoding matrix.

        The positional encoding matrix is defined as follow:

        P_(pos, 2i) = sin(pos / (10000 ^ (2i / d)))
        P_(pos, 2i + 1) = cos(pos / (10000 ^ (2i / d)))

        The output will have shape (batch_size, sequence_length, embedding_dim)
        """

        seq_len = X.size(1)
        d_model = X.size(2)

        positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        even_dims = torch.arange(0, d_model, 2, dtype=torch.float32)
        odd_dims = torch.arange(1, d_model, 2, dtype=torch.float32)

        pos_2d = torch.zeros(seq_len, d_model, device=X.device)

        pos_2d[:, ::2] = torch.sin(positions / (10000 ** (even_dims / d_model)))
        pos_2d[:, 1::2] = torch.cos(positions / (10000 ** ((odd_dims - 1) / d_model)))

        pos_3d = X + pos_2d
        return pos_3d

    def forward(self, data):
        # x: (batch, T, neurons)

        # ---- encoder ----
        x_emb = self.input_proj(data)
        x_emb = self.positional_encoding(x_emb)
        z = self.encoder(x_emb) #, mask=mask)

        # ---- latent ----
        z_latent = self.to_latent(z)
        z_recon = self.from_latent(z_latent)
        out = self.output_proj(z_recon)
        return out, z_latent

def train(model, train_x, train_y, test_x, test_y, lr, epochs, batch_size, weight_decay):
    loss_fn = nn.PoissonNLLLoss(log_input=False)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epoch_train_loss = np.zeros(epochs)
    epoch_test_loss = np.zeros(epochs)
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    for ep in range(epochs):

        train_loss = 0
        test_loss = 0

        #permutation = torch.randperm(train_x.shape[0])
        num_batches = train_x.shape[0] // batch_size
        model.train()

        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for batch_id, (data, target) in enumerate(loader):
            optimizer.zero_grad()
            model_pred, _ = model(data) #, batch_target)

            batch_loss = loss_fn(model_pred, target)#, target[:, 1:])
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        num_test_batches = test_x.shape[0] // batch_size
        model.eval()
        loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for batch_id, (data, target) in enumerate(loader_test): # ba in tqdm(range(num_test_batches), desc="Test", leave=False):
            target_pred, _ = model(data)
            batch_loss = loss_fn(target_pred, target) #[:, 1:])
            test_loss += batch_loss.item()

        epoch_train_loss[ep] = train_loss / num_batches
        epoch_test_loss[ep] = test_loss / num_test_batches
        print(f"Epoch {ep}: Train loss = {epoch_train_loss[ep]:.4f}, Test loss = {epoch_test_loss[ep]:.4f}")
    return epoch_train_loss, epoch_test_loss



if __name__ == "__main__":

    print("Training new transformer encoder :)")
    prep_activity = False
    if prep_activity:
        spikes = np.load("./data/spike_tensor_binned.npy")
        smth_data = np.load("./data/smth_spikes.npy")
    else:
        spikes = np.load("./data/spike_tensor_binned_movement.npy")
        smth_data = np.load("./data/smth_spikes_movement.npy")

    n_trials = smth_data.shape[0]
    n_neurons = smth_data.shape[2]
    print("Number of neurons: ", n_neurons)
    print("Number of time steps: ", smth_data.shape[1])
    print("Total number of trials: ", n_trials)

    n_heads = 6
    n_layers = 3
    eighty_percent = int(np.floor(n_trials * 0.8))
    print("Eighty percent of trials: ", eighty_percent)
    train_x = smth_data[:eighty_percent, :] # smooth data as input to model
    train_y = spikes[:eighty_percent, :]
    test_x = smth_data[eighty_percent:, :]
    test_y = spikes[eighty_percent:, :] # targets as spike counts unsmoothed
    latent_dim_size = 100
    transformer = EncTransformer(n_neurons, d_model=latent_dim_size, n_heads=n_heads, n_layers=n_layers)
    epochs = 20
    batch_size = 64
    lr = 0.0001
    weight_decay = 0.001
    epoch_train_loss, epoch_test_loss = train(transformer, torch.tensor(train_x).to(torch.float32), torch.tensor(train_y).to(torch.float32),
                                              torch.tensor(test_x).to(torch.float32), torch.tensor(test_y).to(torch.float32), lr, epochs, batch_size, weight_decay)

    version = "v1"
    folder_dest = "train_output_movement"
    with open(f'./{folder_dest}/enctransformer_losses_{n_layers}_{n_heads}_{version}.pkl', 'wb') as f:
        pickle.dump((epoch_train_loss, epoch_test_loss), f)
    torch.save(transformer.state_dict(), f'./{folder_dest}/enctransformer_model_{n_layers}lay_{n_heads}heads_{version}.pkl')

    plt.plot([i for i in range(epochs)], epoch_train_loss, label=f'Train Loss (final loss: {epoch_train_loss[-1]:.4f})')
    plt.plot([i for i in range(epochs)], epoch_test_loss, label=f'Test Loss (final loss: {epoch_test_loss[-1]:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f"Transformer Encoder Reconstruction Loss with {n_layers} Layers and {n_heads} heads")
    plt.savefig(f'./{folder_dest}/enctransformer_losses_{n_layers}_{n_heads}_{version}.png')
    plt.close()
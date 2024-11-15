import pickle
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
import torch.optim as optim
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

home_dir = os.path.expanduser("~")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, nhead, num_layers, dropout=0.1):
        super(TransformerEncoderDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Embeddings
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.output_embedding = nn.Linear(hidden_dim, output_dim)

        # Transformer
        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        # Add positional encoding
        src = src + self.positional_encoding[:, :]
        tgt = tgt + self.positional_encoding[:, :]

        # Pass through embedding layers
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        # Apply dropout
        src = self.dropout(src)
        tgt = self.dropout(tgt)

        # Transformer
        output = self.transformer(src, tgt)

        # Pass through output layer
        output = self.output_embedding(output)
        
        return output

# Hyperparameters
input_dim = 409       # Dimension of the input data
output_dim = 409      # Dimension of the output data
hidden_dim = 128      # Hidden dimension in the Transformer model
nhead = 8             # Number of attention heads
num_layers = 4        # Number of Transformer layers
learning_rate = 0.001 # Learning rate
num_epochs = 200       # Number of epochs
batch_size = 32       # Batch size
encoded_dims = [8, 16, 32, 64]

encoded_dim = 64

# Instantiate the model
model = TransformerEncoderDecoder(input_dim, output_dim, encoded_dim, nhead, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()  # For regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Example training data (replace this with your dataset)
# Here we use random tensors for demonstration
# train_data = torch.randn(1000, 409)  # 1000 sequences of length 409
# # train_targets = torch.randn(1000, 409)  # Target data
# train_targets = train_data  # For demonstration purposes, we use the same data as input

def load_data(filename = os.path.join(home_dir, "mod-decision-diffuser/code/diffuser/datasets/data/data_sampled.pkl")):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
        return data['data'], data['info']
    
data, _ = load_data()
data = data['observations']
# data_tensor = torch.tensor(data, dtype=torch.float32)

# train_data, val_data = train_test_split(data_tensor, test_size=0.2, random_state=42)

# train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size, shuffle=False)


def train_and_evaluate(encoded_dim):
    model = TransformerEncoderDecoder(input_dim, output_dim, encoded_dim, nhead, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f'**************Training model with encoded_dim={encoded_dim}')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        # import pdb; pdb.set_trace()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
            sample = batch[0].to(device).unsqueeze(0)  # Move to GPU
            output = model(sample, sample)
            loss = criterion(output, sample)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            optimizer.step()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
        
        # Validate every 25 epochs
        if (epoch + 1) % 25 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    sample = batch[0].to(device)  # Move to GPU
                    output = model(sample, sample)
                    loss = criterion(output, sample)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            print(f'Validation Loss after Epoch [{epoch+1}/{num_epochs}]: {val_loss:.4f}')
            
            # Save the model checkpoint
            checkpoint_filename = f'final_checkpoint_epoch_{epoch+1}_encoded_dim_{encoded_dim}.pth'
            torch.save(model.state_dict(), checkpoint_filename)
            print(f"checkpoint saved! {checkpoint_filename}")
            
            model.train()  # Switch back to training mode
        # Check gradients
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f'Gradient for {name}: {param.grad.norm().item()}')

    print(f'**************Finished training model with encoded_dim={encoded_dim}')
    print(f'the training loss: {avg_epoch_loss:.4f}')

    # return avg_epoch_loss

    # Evaluation on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            sample = batch[0].to(device)  # Move to GPU
            output = model(sample, sample)
            loss = criterion(output, sample)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # # Save the model
    model_filename = f'autoencoder_encoded_dim_{encoded_dim}.pth'
    torch.save(model.state_dict(), model_filename)
    
    return avg_epoch_loss, val_loss

def load_encode_model(encoded_dim, checkpoint_path):
    model = TransformerEncoderDecoder(input_dim, output_dim, encoded_dim, nhead, num_layers).to(device)
    try:
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            model.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find checkpoint file at {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading encode model checkpoint: {str(e)}")
    # model.eval()  # Set the model to evaluation mode
    return model

def save_encode_model(model, encoded_dim, step, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save full model
    model_path = os.path.join(save_dir, f'encode_model_epoch_{step}_dim_{encoded_dim}.pth')
    torch.save(model.state_dict(), model_path)
    

def encode_data(model, obs_data):
    if type(obs_data) == dict:
        data = obs_data[0]
    else:
        data = obs_data
    data += model.positional_encoding[:, :]
    # Pass through embedding layer
    embedded_data = model.embedding(data)
    # Pass through transformer encoder
    encoded_data = model.transformer.encoder(embedded_data)
    if type(obs_data) == dict:
        obs_data[0] = encoded_data
        return obs_data
    else:
        return encoded_data


def tb_encode_data(model, obs_data): # madiff_clone env 
    # Create a SummaryWriter instance
    writer = SummaryWriter('runs/encoder_graph')
    
    # Convert input to torch tensor if needed
    if not isinstance(obs_data, torch.Tensor):
        obs_data = torch.tensor(obs_data, dtype=torch.float32)
    
    if type(obs_data) == dict:
        data = obs_data[0]
    else:
        data = obs_data
        
    # Add a batch dimension if missing
    if len(data.shape) == 2:
        data = data.unsqueeze(0)

    # Move data to the same device as the model
    data = data.to(device)
    
    def forward_pass(data):
        data = data + model.positional_encoding[:, :]
        embedded_data = model.embedding(data)
        encoded_data = model.transformer.encoder(embedded_data)
        return encoded_data
    
    # Add graph to tensorboard
    writer.add_graph(model, (data, data))  # Adding both src and tgt tensors
    writer.close()
    
    # Perform the actual encoding
    with torch.no_grad():
        encoded_data = forward_pass(data)
    
    if type(obs_data) == dict:
        obs_data[0] = encoded_data
        return obs_data
    else:
        return encoded_data.squeeze(0) if len(data.shape) == 3 else encoded_data


def encode_data_batch(model, data):
    data = np.array(data)
    data = torch.tensor(data, dtype=torch.float32)
    data_loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    print("encoding!!!")
    results = torch.tensor([]).to(device)

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('runs1/encod_data')

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Encoding data', leave=False):
            batch = batch[0].to(device)
            # Add positional encoding
            batch += model.positional_encoding[:, :]
            # Pass through embedding layer
            embedded_data = model.embedding(batch)
            # Pass through transformer encoder
            encoded_data = model.transformer.encoder(embedded_data)
            results = torch.cat((results, encoded_data), 0)

            # Modify this line
            # writer.add_graph(model, (batch, batch))

    # writer.close()
    return results

# # Example usage
epoch = 200  # Use the best epoch found during training
# encoded_dim = 64 # Use the best encoded_dim found during training
# home_dir = os.path.expanduser("~")

model = load_encode_model(encoded_dim, f'final_checkpoint_epoch_{epoch}_encoded_dim_{encoded_dim}.pth')
 
# data, _ = load_data()
# data = data['observations']

encoded_data = tb_encode_data(model, data)

# print(encoded_data.shape)

# # Save encoded_data
# save_path = os.path.join(home_dir, f"grid-diffuser/code/diffuser/datasets/data/encoded_data_dim_{encoded_dim}.pt")
# torch.save(encoded_data, save_path)
# print(f"Encoded data saved to: {save_path}")

def load_encoded_data(encoded_dim):
    home_dir = os.path.expanduser("~")
    load_path = os.path.join(home_dir, f"grid-diffuser/code/diffuser/datasets/data/encoded_data_dim_{encoded_dim}.pt")
    
    if os.path.exists(load_path):
        encoded_data = torch.load(load_path)
        print(f"Encoded data loaded from: {load_path}")
        print(f"Encoded data shape: {encoded_data.shape}")
        return encoded_data
    else:
        print(f"No encoded data found at: {load_path}")
        return None

# # Example usage
# encoded_dim = 64  # Use the same dimension as used for encoding
# loaded_encoded_data = load_encoded_data(encoded_dim)

# if loaded_encoded_data is not None:
#     # You can now use the loaded_encoded_data for further processing or analysis
#     pass





import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml

class FCMT(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, d_ff, dropout=0.1):
        super(FCMT, self).__init__()
        
        # 1x1 Convolution for item-level map learning
        self.conv1x1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        
        # Transformer encoder block with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Linear projection to match d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # Input x: [batch_size, sequence_length, input_dim]
        x = self.input_projection(x)  # Project input to d_model dimension
        
        # Add positional encoding
        x = self.positional_encoding(x)

        # Pass through transformer encoder
        transformed_x = self.transformer_encoder(x)

        # Compute final output
        y_L = self.layer_norm(transformed_x)
        z = y_L + transformed_x  # Lambda = 1

        return z

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

if __name__ == "__main__":
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model = FCMT(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    ).cuda()
    
    sample_input = torch.rand(2, 10, config['input_dim']).cuda()
    output = model(sample_input)
    print("Output shape:", output.shape)

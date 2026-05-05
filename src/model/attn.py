import torch.nn as nn

class AttnMLP:
    def __init__(self, d_model, hidden_dim, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # layer 1
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.act = nn.GeLU()
        self.do1 = nn.Dropout(dropout)

        # layer 2 
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.do2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x

        output = self.fc1(x)
        output = self.act(output)
        output = self.do1(output)
        output = self.fc2(output)
        
        output = output + res 

        output = self.layer_norm(output)

class MultiHeadAttn:
    ...
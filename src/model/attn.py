import torch
import torch.nn as nn
import numpy as np

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
    def __init__(
            self, 
            d_model,
            num_heads,
            dropout=0.3
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attn(self, Q, K, V, mask=None):
        d_k = self.head_dim

        # compute attn
        attn_scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # apply mask
        if mask is not None:
            attn_scores = attn_scores + mask

        # softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # multiply by V
        output = attn_weights @ V

        return output, attn_weights
    
    def split_heads(self, x):
        """
        """
        batch_size, seq_len, d_model = x.shape
        
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)

        return x
    
    def forward(self, query, key, value, mask=None):
        """
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        res = query

        Q = self.Wq(query)
        K = self.Wk(key)
        V = self.Wv(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attn_output, attn_weights = self.scaled_dot_product_attn(Q, K, V, mask)

        attn_output= attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(batch_size, -1, self.d_model)

        output = self.Wo(attn_output)

        output = output + res
        output = self.layer_norm(output)

        return output, attn_weights
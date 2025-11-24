import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFusionModel(nn.Module):
    def __init__(self, feature_dim=384, hidden_dims=[1024,512,256], dropout_rate=0.2):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate

        layers = []
        
        prev_dim = feature_dim*9
        for dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU()) 
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim=dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, socio_demo_tensor, text_tensor):
        batch_size = socio_demo_tensor.shape[0]

        # text tensor dim: [batch, 384]
        
        text_features = text_tensor.squeeze(1) if text_tensor.dim() == 3 else text_tensor
        # socio-tensor dim: [batch, 8*384]
        socio_flattened = socio_demo_tensor.view(batch_size, -1)

        # concate [batch, 8*384 + 384] = [batch, 9*384]
        combined_features = torch.cat([socio_flattened, text_features], dim=1)

        # fusion [batch, 384]
        probabilities = self.network(combined_features)

        # shape: [batch]
        return probabilities.squeeze(-1)


class CrossAttentionModel(nn.Module):
    
    def __init__(self, feature_dim=384, hidden_dim=128, dropout_rate=0.1):
        super().__init__()
        self.feature_dim = feature_dim

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.attention_norm = nn.LayerNorm(feature_dim)
        self.attention_dropout = nn.Dropout(dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, socio_demo_tensor, text_tensor):
        batch_size = socio_demo_tensor.shape[0]

        if text_tensor.dim() == 3 and text_tensor.size(1) > 1:
            text_tensor = text_tensor.mean(dim=1, keepdim=True)

        if text_tensor.dim() == 2:
            text_tensor = text_tensor.unsqueeze(1)

        Q = self.query_proj(text_tensor)
        K = self.key_proj(socio_demo_tensor)
        V = self.value_proj(socio_demo_tensor)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (8.0 ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_features = torch.matmul(attention_weights, V)
        attended_features_flat = attended_features.squeeze(1)
        text_tensor_flat = text_tensor.squeeze(1)

        combined_features = torch.cat([attended_features_flat, text_tensor_flat], dim=1)
        logits = self.classifier(combined_features)

        return logits.squeeze(-1), attention_weights.squeeze(1)
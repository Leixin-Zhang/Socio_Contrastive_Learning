import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrasive_Model(nn.Module):
    def __init__(self, socio_dim=51, other_dim=384, embed_dim=[64,128], hidden_dims=[512, 256], contrastive_dim=128):
        super().__init__()

        # Socio-demographic encoder (用于对比学习)
        self.socio_encoder_1 = nn.Linear(socio_dim, embed_dim[0])
        self.socio_encoder_2 = nn.Linear(embed_dim[0], embed_dim[1])

        # Contrastive projection - merely based on socio-demographic
        self.contrastive_projection = nn.Linear(embed_dim[1], contrastive_dim)

        # Main network layers
        shared_dims = [embed_dim[1] + other_dim] + hidden_dims
        self.shared_layers = nn.ModuleList()
        for i in range(len(shared_dims)-1):
            self.shared_layers.append(nn.Linear(shared_dims[i], shared_dims[i+1]))

        # Output layer
        self.output = nn.Linear(hidden_dims[-1], 1)

        # Regularization
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(embed_dim[1] + other_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dims[0])

    def forward(self, socio ,other, return_contrastive=False):
        
        # Encode socio-demographic features
        socio_embed = F.leaky_relu(self.socio_encoder_1(socio))
        socio_embed = F.leaky_relu(self.socio_encoder_2(socio_embed))

        # 计算对比学习嵌入 - 只基于socio特征
        if return_contrastive:
            contrastive_emb = self.contrastive_projection(socio_embed)
            contrastive_emb = F.normalize(contrastive_emb, p=2, dim=1)

        # Main network path
        x = torch.cat([socio_embed, other], dim=1)
        x = self.bn1(x)

        for i, layer in enumerate(self.shared_layers):
            x = F.leaky_relu(layer(x))
            if i == 0:
                x = self.bn2(x)
            x = self.dropout(x)

        classification_output = self.output(x)

        if return_contrastive:
            return classification_output, contrastive_emb
        else:
            return classification_output



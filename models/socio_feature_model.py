import torch
import torch.nn as nn

class Socio_Feature_Model(nn.Module):
    def __init__(self, input_dim=435, hidden_dims=[512,256], dropout_rate=0.2):
        super(Socio_Feature_Model, self).__init__()

        layers = []
        prev_dim = input_dim

        self.input_norm = nn.BatchNorm1d(input_dim)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, social, text):

        x = torch.concat([social,text], axis=1)
        x = self.input_norm(x)

        return self.network(x).squeeze()
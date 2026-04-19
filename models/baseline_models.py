import torch
import torch.nn as nn

class Simple_Model(nn.Module):
    def __init__(self, input_dim=384, hidden_dims=[512,256], dropout=0.2):
        super(Simple_Model, self).__init__()

        layers = []
        prev_dim = input_dim

        self.input_norm = nn.BatchNorm1d(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_norm(x)
        return self.network(x).squeeze()


class Multi_Task_Model(nn.Module):
    def __init__(self, input_dim=384, num_annotators=2316, hidden_dim=[512,216]):
        super().__init__()
        self.num_annotators = num_annotators

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 10),
            nn.ReLU(),
        )

        self.task_heads = nn.ModuleList([
            nn.Linear(10, 1) for _ in range(self.num_annotators)
        ])

    def forward(self, x):
        shared_features = self.shared_encoder(x)

        outputs = []
        for head in self.task_heads:
            outputs.append(head(shared_features).squeeze(-1))

        return torch.stack(outputs, dim=1)
    




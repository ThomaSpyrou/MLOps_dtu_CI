from torch import nn


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, latent_dim):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')

        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
            
        return self.model(x)
import torch
import torch.nn as nn

from src.core.config import GRUArgs


class GRU(nn.Module):
    def __init__(self, config: GRUArgs) -> None:
        super(GRU, self).__init__()
        
        self.n_layers = config.n_layers
        self.hidden_size = config.hidden_size

        # RNN
        self.gru = nn.GRU(input_size=config.input_size, 
                         hidden_size=config.hidden_size,
                         num_layers=config.n_layers,
                         batch_first=True,
                         dropout=config.dropout)
        # Linear 
        self.fc = nn.Linear(config.hidden_size, config.n_classes)
        

    def forward(self, x, device):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h_0)
        out = out[:, -1, :] # para obter apenas a ultima saida da rnn
        out = self.fc(out)

        return out




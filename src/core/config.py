from dataclasses import dataclass

@dataclass()
class RNNArgs:
    input_size: int = 28
    sequence_size: int = 28
    n_layers: int = 2 # quantidade de rnn que serao empilhadas
    hidden_size: float = 256
    n_classes: int = 10
    dropout: bool = 0

@dataclass()
class GRUArgs:
    input_size: int = 28
    sequence_size: int = 28
    n_layers: int = 2 # quantidade de rnn que serao empilhadas
    hidden_size: float = 256
    n_classes: int = 10
    dropout: bool = 0

@dataclass()
class LSTMArgs:
    input_size: int = 28
    sequence_size: int = 28
    n_layers: int = 2 # quantidade de rnn que serao empilhadas
    hidden_size: float = 256
    n_classes: int = 10
    dropout: bool = 0

@dataclass()
class HyperParameters:
    n_epochs: int = 20
    batch_size: int = 128
    lr: float = 0.001
    num_workers: int = 8
    dir_base:str = "./weights"

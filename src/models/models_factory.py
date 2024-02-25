

from src.core.config import GRUArgs, LSTMArgs, RNNArgs
from src.core.model_types import ModelTypes
from src.models.gru import GRU
from src.models.rnn import RNN


class ModelsFactory:
    
    @staticmethod
    def get_instance(model_type: ModelTypes, device="cuda"):

        # padrao
        model_name = "RNN"
        config = RNNArgs() # Caso voce precise ajustar algum parametro...
        model = RNN(config=config).to(device=device)
        
        if model_type == ModelTypes.LSTM:
            model_name = "LSTM"
            config = LSTMArgs()
        
        if model_type == ModelTypes.GRU:
            model_name = "GRU"
            config = GRUArgs()
            model = GRU(config=config).to(device=device)

        
        return model_name, config, model 


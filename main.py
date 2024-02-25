'''
RNN - Recurrent neural networks
Autor: Adriano A. Santos

Exemplo da utilizacao de uma RNN em Visao Computacional

Modelos:

- RNN
- GRU
- LSTM

'''
import torch
from src.core.config import HyperParameters
from src.core.helper import save_weights
from src.core.model_types import ModelTypes
from src.data.dataloaders import MNISTDataloarder
from src.data.dataset import MNISTDataset
from src.data.transformers import MNISTTransformers
from src.models.models_factory import ModelsFactory
from src.train import training
from src.validation import validation

import argparse

def main(parser):
    # Limpando cache do CUDA
    torch.cuda.empty_cache()

    # Parametro de otmizacao de GPU
    torch.backends.cudnn.benchmark = True

    # Parametros
    author = "Adriano A. Santos"
    hyper_parameters = HyperParameters()
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # *************************** Datasets *************************** 

    ## Transforms
    training_transformer = MNISTTransformers(True).get_instance()
    validation_transformer = MNISTTransformers(False).get_instance()

    ## Datasets
    train_dataset = MNISTDataset("./src/data/files", True, training_transformer).get_instance()
    validation_dataset = MNISTDataset("./src/data/files", False, validation_transformer).get_instance()

    ## Dataloaders
    train_loader = MNISTDataloarder(dataset=train_dataset,batch_size=hyper_parameters.batch_size, shuffle=True, num_worker=hyper_parameters.num_workers ).get_instance()
    validation_loader = MNISTDataloarder(dataset=validation_dataset,batch_size=hyper_parameters.batch_size, num_worker=hyper_parameters.num_workers).get_instance()

    # Cria instancia do modelo
    model_type = ModelTypes[parser.model_type.upper()]
    model_name, model_config, model  = ModelsFactory.get_instance(model_type)

    # Parametros de treinamento
    criterion = torch.nn.CrossEntropyLoss()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=hyper_parameters.lr)

    # Variavel auxiliar
    best_acc = 0

    # Descricao
    print("Treinamento de um modelo com a arquitetura {} em {} epocas.".format(model_name, hyper_parameters.n_epochs))

    # Treinamento e Validacao
    for epoch in range(hyper_parameters.n_epochs):
        print(' \n *********** Epoch {} *********** \n'.format(int(epoch)+1))
        
        # Treinamento
        training(model, criterion, optimizer, train_loader, model_config, device)
        
        # Validacao
        acc_current = validation(model, validation_loader, model_config, device)
        
        # Salvando o modelo
        if acc_current > best_acc:
            # Salva o melhor modelo de acordo com a acuracia
            save_weights(author, model, model_name, hyper_parameters.dir_base, epoch, best_acc)
            best_acc = acc_current

        # Liberando o cache do CUDA
        torch.cuda.empty_cache()

    print("\n Using the arc {} the best acc in {} epochs was {}.".format(model_name, hyper_parameters.n_epochs, best_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="rnn", choices=["rnn", "gru", "lstm"])
    parser = parser.parse_args()
    
    main(parser)
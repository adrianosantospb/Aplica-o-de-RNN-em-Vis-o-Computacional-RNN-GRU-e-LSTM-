import torch
import torchvision.transforms as transforms

class MNISTTransformers:
    
    def __init__(self, train) -> None:
        super(MNISTTransformers, self).__init__()
        self.transformer = self.__get_trasformer(train)

    def get_instance(self):
        return self.transformer

    # Caso voce queira adicionar novas transformacoes.
    def __get_trasformer(self,train):
        if train:
            return transforms.Compose([
                transforms.ToTensor(),
            ])
        return transforms.Compose([
                transforms.ToTensor(),
            ]) 


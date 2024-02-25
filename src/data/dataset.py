from pathlib import Path
import torchvision

class MNISTDataset:
    def __init__(self, root, train, transforms) -> None:
        super(MNISTDataset, self).__init__()
        download = self.__directory_is_empty(root) 
        self.dataset = torchvision.datasets.MNIST(root=root, train=train, transform=transforms, download=download)
    
    def get_instance(self):
        return self.dataset
    
    def __len__(self):
        return len(self.dataset)

    def __directory_is_empty(self, directory: str) -> bool:
        return not any(Path(directory).iterdir())
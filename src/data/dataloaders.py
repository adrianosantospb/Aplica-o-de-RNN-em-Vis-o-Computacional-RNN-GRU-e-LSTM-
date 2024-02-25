from torch.utils.data.dataloader import DataLoader

class MNISTDataloarder:
    def __init__(self, dataset, batch_size, shuffle=False, num_worker=8) -> None:
        self.dataloader = DataLoader(dataset=dataset, 
                                     batch_size=batch_size, 
                                     shuffle=shuffle,
                                     num_workers=num_worker, 
                                     pin_memory=True)
    
    def get_instance(self):
        return self.dataloader
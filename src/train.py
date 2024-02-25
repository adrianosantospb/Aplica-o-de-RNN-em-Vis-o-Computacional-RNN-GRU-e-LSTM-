from tqdm import tqdm

def training(model, criterion, optimizer, dataloader, conf, device):
    
    print('Training step')
    model.train()
    
    for _, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):

        images = images.reshape(-1, conf.sequence_size, conf.input_size)
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images, device)
        
        loss = criterion(output, labels)

        # Backward and optimize
        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()
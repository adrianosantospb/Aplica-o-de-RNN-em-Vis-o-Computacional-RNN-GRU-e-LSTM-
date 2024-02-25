import torch

def validation(model, validation_loader, conf, device):
    
    print('Validation step')

    model.eval() 
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in validation_loader:
            images = images.reshape(-1, conf.sequence_size, conf.input_size).to(device)
            labels = labels.to(device)
            outputs = model(images, device)
            # best (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print('Accuracy: {} %'.format(acc))
    return acc
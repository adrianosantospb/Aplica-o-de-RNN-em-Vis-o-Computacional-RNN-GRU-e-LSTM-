import torch

def save_weights(author, model, model_name, dir_base, epoch, best_acc):
    print("New best model.")
    # Save the best model
    weights = "{}/{}.pt".format(dir_base,str("best") + model_name)
    chkpt = __get_checkpoint (author, model, epoch, best_acc) 
    torch.save(chkpt, weights)

def __get_checkpoint(author, model, epoch, best_acc):
    chkpt = {'author': author, 'epoch': epoch,'model': model.state_dict(), "best_acc": best_acc}
    return chkpt
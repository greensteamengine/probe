from torch import nn
import torch as pt
import data_handler as dh
from torch.utils.data import Dataset, DataLoader
import time


#indpired and by and used some code from https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb
class data_set(Dataset):
    def __init__(self, data_dict, target_dict):
        data, targets = [], []
        for i in range(len(data_dict)):
        #for i in range(20):
        
            #print(data_dict[i])
            #print(target_dict[i][2])
            data.append(data_dict[i].view(1,768))
            targets.append(target_dict[i][2].item())
            
        self.data = pt.cat(data)
        self.tergets = pt.tensor(targets)
  
    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, index):
        return self.data[index], self.tergets[index]

class MLP(nn.Module):
    def __init__(self, hid_size=50):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 37)
        )
        
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc, top_pred.view_as(y)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_run(model, dataloader, optimizer, loss_function):

    # Set current loss value
    epoch_loss = 0
    epoch_acc = 0
    
    CE = 0 #classification error
    #RE = 0 #regression error ... <- sum Cost / loss / error function
    
    # Iterate over the DataLoader for training data
    for i, data in enumerate(dataloader, 0):
        
        # Get inputs
        inputs, targets = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        acc, _ = calculate_accuracy(outputs, targets)
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print statistics
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)
    
def update_confusion(confusion_matrix, targets, predictions):
    for i, t in enumerate(targets):
        confusion_matrix[t][predictions[i]] += 1 

def evaluate_run(model, dataloader, loss_function):
    
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    
    confusion_matrix = pt.zeros([37, 37])
    
    with pt.no_grad():

        for i, data in enumerate(dataloader, 0):

            inputs, targets = data
            outputs = model(inputs)
            
            

            loss = loss_function(outputs, targets)

            acc, predictions = calculate_accuracy(outputs, targets)

            #print(f"targets: {targets}")
            #print(f"predictions: {predictions}")
            
            update_confusion(confusion_matrix, targets, predictions)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), confusion_matrix

def train(model, train_loader, eval_loader, optimizer, loss_function, EPOCHS):
        

    best_valid_loss = float('inf')
    
    epoch_times = []
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    for epoch in range(EPOCHS):

        start_time = time.monotonic()

        train_loss, train_acc = train_run(model, train_loader, optimizer, loss_function)
        valid_loss, valid_acc, _ = evaluate_run(model, eval_loader, loss_function)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            pt.save(model.state_dict(), 'tut1-model.pt')

        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        epoch_times.append(end_time - start_time)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        
    return train_losses, train_accs, valid_losses, valid_accs, epoch_times



def probe_layer(layer_num, EPOCHS, batch_size, hid_size):
    
    large_meta = dh.read_tags_from_dt("train_data")
    large_states = dh.read_states_from_dt("train_data", layer_num)
    
    
    
    train_meta, eval_meta = dh.separate_dict(large_meta, 0.8)
    train_states, eval_states = dh.separate_dict(large_states, 0.8)

    del large_meta
    del large_states
    
    #train_meta, eval_meta = dh.separate_dict(test_meta, 0.8)
    #train_states, eval_states = dh.separate_dict(test_states, 0.8)
    
    #WARNING TODO test -> train
    train_data = data_set(train_states, train_meta)
    eval_data = data_set(eval_states, eval_meta)
    
    
    train_loader = pt.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    eval_loader = pt.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=True, num_workers=1)
    
    
    model = MLP(hid_size)
    
    #https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md
    
    # Define the loss function and optimizer
    #https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_function = nn.CrossEntropyLoss()
    optimizer = pt.optim.Adam(model.parameters(), lr=1e-4)

    #train(mlp, trainloader, optimizer, loss_function)
    print(f"layer: {layer_num} epochs: {EPOCHS} batch size: {batch_size} hidden layer size: {hid_size}")
    print("TRAINING")
    train_losses, train_accs, valid_losses, valid_accs, epoch_times = train(model, train_loader, eval_loader, optimizer, loss_function, EPOCHS)
    
    train_meta_res = (train_losses, train_accs, valid_losses, valid_accs, epoch_times)
    
    print(f"train_losses: {train_losses}")
    print(f"train_accs: {train_accs}")
    print(f"valid_losses: {valid_losses}")
    print(f"valid_accs: {valid_accs}")
    print(f"epoch_times: {epoch_times}")
    
    del train_loader
    del eval_loader
    
    del train_data
    del eval_data
    
    del train_meta
    del eval_meta
    del train_states
    del eval_states
    
    print("TESTING")
    
    
    test_meta = dh.read_tags_from_dt("test_data")
    test_states = dh.read_states_from_dt("test_data", layer_num)
    test_data = data_set(test_states, test_meta)
    
    test_loader = pt.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
    
    test_loss, test_acc, confusion_matrix = evaluate_run(model, test_loader, loss_function)
    
    print(f'\t Layer tested: {layer_num} -> Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
    
    del test_meta
    del test_states
    del test_data
    del test_loader
    
    return test_loss, test_acc, confusion_matrix, train_meta_res

if __name__ == '__main__':
    pt.manual_seed(77)
    
    #for cur_layer in range(13):
    """
    cur_layer = 6
    large_meta = dh.read_tags_from_dt("train_data")
    large_states = dh.read_states_from_dt("train_data", cur_layer)
    
    
    
    train_meta, eval_meta = dh.separate_dict(large_meta, 0.8, 100)
    train_states, eval_states = dh.separate_dict(large_states, 0.8, 100)

    del large_meta
    del large_states
    
    #train_meta, eval_meta = dh.separate_dict(test_meta, 0.8)
    #train_states, eval_states = dh.separate_dict(test_states, 0.8)
    
    #WARNING TODO test -> train
    train_data = data_set(train_states, train_meta)
    eval_data = data_set(eval_states, eval_meta)
    
    
    train_loader = pt.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1)
    eval_loader = pt.utils.data.DataLoader(eval_data, batch_size=100, shuffle=True, num_workers=1)
    
    
    model = MLP()
    
    #https://github.com/christianversloot/machine-learning-articles/blob/main/creating-a-multilayer-perceptron-with-pytorch-and-lightning.md
    
    # Define the loss function and optimizer
    #https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    loss_function = nn.CrossEntropyLoss()
    optimizer = pt.optim.Adam(model.parameters(), lr=1e-4)

    #train(mlp, trainloader, optimizer, loss_function)
    EPOCHS = 5

    train(model, train_loader, eval_loader, optimizer, loss_function, EPOCHS)
    
    del train_loader
    del eval_loader
    
    del train_data
    del eval_data
    
    del train_meta
    del eval_meta
    del train_states
    del eval_states
    
    print("TESTING")
    
    
    test_meta = dh.read_tags_from_dt("test_data")
    test_states = dh.read_states_from_dt("test_data", cur_layer)
    test_data = data_set(test_states, test_meta)
    
    test_loader = pt.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
    
    test_loss, test_acc = evaluate_run(model, test_loader, loss_function)
    
    print(f'\t Layer tested: {cur_layer} -> Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
    
    del test_meta
    del test_states
    del test_data
    del test_loader
    del test_loss
    del test_acc
    """
    
    epochs = [1,3,5]
    batch_sizes = [10, 100, 500]
    hid_size = [50, 70]
    
    
    for e in epochs:
        for b in batch_sizes:
            for h in hid_size:
                #layer six was the most sccessful
                res = probe_layer(6, e, b, h)
                print(res)
                conf_matr = res[2]
                #dh.print_conf_matrix(conf_matr)
    

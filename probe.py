from torch import nn
import * from data_handler

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 37)
        )
        
    def forward(self, x):
        #x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
    
    
if __name__ == '__main__':
    torch.manual_seed(77)
    
    train_meta = data_handler.read_tags_from_dt("train_data")
    train_states = data_handler.read_states_from_dt("train_data", 1)
    
    test_meta = data_handler.read_tags_from_dt("test_data")
    test_states = data_handler.read_states_from_dt("test_data", 1)
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

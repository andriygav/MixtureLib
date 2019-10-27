import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

class HyperModel:
    def __init__(self):
        pass

    def E_step(self, X, Y, Z, HyperParameters):
        raise NotImplementedError

    def M_step(self, X, Y, Z, HyperParameters):
        raise NotImplementedError

    def LogPiExpectation(self, X, Y, HyperParameters):
        raise NotImplementedError

    def PredictPi(self, X, HyperParameters):
        raise NotImplementedError   


class HyperModelDirichlet(HyperModel):
    def __init__(self, output_dim = 2, device = 'cpu'):
        super(HyperModelDirichlet, self).__init__()
        self.output_dim = output_dim
        self.device = device
        
        self.mu = torch.ones(self.output_dim)
        self.m = torch.zeros_like(self.mu)
        self.m.data = self.mu.data.clone()
        self.N = 0
    
    def E_step(self, X, Y, Z, HyperParameters):
        gamma = Z.sum(dim=0)
        self.m = (self.mu + gamma).detach()
        self.N = Z.shape[0]
        pass
    
    def M_step(self, X, Y, Z, HyperParameters):
        pass

    def LogPiExpectation(self, X, Y, HyperParameters):
        return torch.ones_like(X)*(torch.digamma(self.m) - torch.digamma(self.output_dim*self.mu + self.N))

    def PredictPi(self, X, HyperParameters):
        return torch.ones_like(X)*torch.nn.functional.softmax(self.LogPiExpectation(X, None, HyperParameters), dim = -1)


class HyperExpertNN(nn.Module, HyperModel):
    def __init__(self, input_dim = 20, hidden_dim = 10, output_dim = 10, epochs=100, device = 'cpu'):
        super(HyperExpertNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.epochs=epochs
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.0001)
        
        self.to(device)
        
    def forward(self, input):
        out = input
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out
    
    def E_step(self, X, Y, Z, HyperParameters):
        pass
    
    def M_step(self, X, Y, Z, HyperParameters):
        dataset = TensorDataset(X.to(self.device), Z.to(self.device))
        
        for _ in range(self.epochs):
            train_generator = DataLoader(dataset = dataset, batch_size = 128, shuffle=True)
            for it, (batch_of_x, batch_of_z) in enumerate(train_generator):
                self.zero_grad()
                loss = -(torch.nn.functional.log_softmax(self.forward(batch_of_x), dim = -1)*batch_of_z).mean()
                loss.backward()
                self.optimizer.step()
        pass

    def LogPiExpectation(self, X, Y, HyperParameters):
        return torch.nn.functional.log_softmax(self.forward(X), dim = -1)
    
    def PredictPi(self, X, HyperParameters):
        return torch.nn.functional.softmax(self.forward(X), dim = -1)


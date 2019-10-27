import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import math

class TorchDataset(Dataset):
    def __init__(self, X, Y, device = 'cpu'):
        super(TorchDataset, self).__init__()
        self.X = torch.FloatTensor(X).to(device)
        self.Y = torch.FloatTensor(Y).to(device)
        self.len = X.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class HyperModelNN(nn.Module):
    def __init__(self, input_dim = 20, hidden_dim = 10, output_dim = 10, epochs=100, device = 'cpu'):
        super(HyperModelNN, self).__init__()
        
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
        dataset = TorchDataset(X, Z, device = self.device)
        
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

class EachModelLinear:
    def __init__(self, input_dim = 20, device = 'cpu', A = None, w = None, regul = True):
        self.input_dim = input_dim
        self.device = device
        
        self.regul = regul
        self.A = A
            
        self.W = (1e-5)*torch.randn(input_dim, 1, device = self.device)
        
        if w is not None:
            self.w_0 = w.clone()
            self.W.data = w.data.clone() + (1e-5)*torch.randn(input_dim, 1, device = self.device)
        else:
            self.w_0 = w
        
        self.B = torch.eye(input_dim, device = self.device)
        if self.A is not None:
            if len(self.A.shape) == 1:
                self.B.data = torch.diag(self.A).data.clone()
            else:
                self.B.data = self.A.data.clone()
        
    def forward(self, input):
        return input@self.W
    
    def __call__(self, input):
        return self.forward(input)
    
    def OptimizeHyperParameters(self, X, Y, Z, HyperParameters, Parameter):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        Z is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        Parameter is a Key in dictionary
        """
        if Parameter == 'beta':
            temp1 = Y**2
            temp2 = -2*Y*(X@self.W)
            temp3 = torch.diagonal(X@(self.B+self.W@self.W.transpose(0,1))@X.transpose(0,1)).view([-1, 1])
            return ((temp1 + temp2 + temp3)*Z).mean().detach()
        
    def LogLikeLihoodExpectation(self, X, Y, HyperParameters):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        """
        beta = 1./(HyperParameters['beta'] + 0.000001)
        temp1 = Y**2
        temp2 = -2*Y*(X@self.W)
        temp3 = torch.diagonal(X@(self.B+self.W@self.W.transpose(0,1))@X.transpose(0,1)).view([-1, 1])
        return (-0.5*beta*(temp1 + temp2 + temp3) + 0.5*math.log(beta/(2*math.pi))).detach()
        

    def E_step(self, X, Y, Z, HyperParameters):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        Z is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        """
        beta = 1./(HyperParameters['beta'] + 0.000001)
        temp = X.unsqueeze(2)
        
        if self.A is None:
            self.B = torch.inverse(((temp*Z.unsqueeze(1))@temp.transpose(2, 1)).sum(dim = 0)).detach()
            second = (X*Y*Z).sum(dim = 0).view([-1, 1])
            self.W.data = (self.B@second).view_as(self.W).detach()
        else:
            A = self.A
            if len(self.A.shape) == 1:
                A = torch.diag(self.A)
            
            self.B = torch.inverse(torch.inverse(A) + beta*((temp*Z.unsqueeze(1))@temp.transpose(2, 1)).sum(dim = 0)).detach()
            second = beta*(X*Y*Z).sum(dim = 0).view([-1, 1])       
            if self.w_0 is None:
                self.W.data = ((self.B@second)).view_as(self.W).detach()
            else:
                self.W.data = (self.B@(second + torch.inverse(A)@self.w_0)).view_as(self.W).detach()
        
        return

    def M_step(self, X, Y, Z, HyperParameters):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        Z is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        """
        beta = 1./(HyperParameters['beta'] + 0.000001)
        
        if self.A is not None:
            if self.w_0 is not None:
                if len(self.A.shape) == 1:
                    self.A= torch.diagonal(self.B+self.W@self.W.transpose(0,1) - self.w_0@self.W.transpose(0,1) - self.W@self.w_0.transpose(0,1) + self.w_0@self.w_0.transpose(0,1)).detach()
                else:
                    self.A= (self.B+self.W@self.W.transpose(0,1) - self.w_0@self.W.transpose(0,1) - self.W@self.w_0.transpose(0,1) + self.w_0@self.w_0.transpose(0,1)).detach()
            else:
                if len(self.A.shape) == 1:
                    self.A = torch.diagonal(self.B+self.W@self.W.transpose(0,1)).detach()
                else:
                    self.A = (self.B+self.W@self.W.transpose(0,1)).detach()
                
        
        if self.w_0 is not None:
            if self.regul == True:
                self.w_0.data[2:,:] = self.W.data[2:,:].clone()
            else:
                self.w_0.data = self.W.data.clone()

        return


class RegularizeModel:
    def __init__(self, ListOfModels = None, device = 'cpu'):
                
        if ListOfModels is None:
            self.ListOfModels = []
        else:
            self.ListOfModels = ListOfModels

        
    def forward(self, input):
        pass
    
    def __call__(self, input):
        pass

    def E_step(self, X, Y, Z, HyperParameters):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        Z is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        """
        pass

    def M_step(self, X, Y, Z, HyperParameters):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        Z is a tensor of shape [N x K]
        HyperParameters is a dictionary
        """
        alpha = 1./(HyperParameters['alpha']+1e-30)
        beta = 1./(HyperParameters['beta'] + 0.000001)
        K = len(self.ListOfModels)
        
        ListOfNewW0 = []
        
        for k in range(K):
            if self.ListOfModels[k].w_0 is not None:
                if len(self.ListOfModels[k].A.shape) == 1:
                    A_inv = torch.diag(1./self.ListOfModels[k].A)
                else:
                    A_inv = torch.inverse(self.ListOfModels[k].A)
                
                B = self.ListOfModels[k].B

                temp1 = torch.inverse(A_inv \
                                        +alpha*(K)*torch.diag(torch.ones_like(self.ListOfModels[k].w_0.view(-1))))
                temp2 = A_inv@self.ListOfModels[k].W \
                        + alpha*torch.cat([self.ListOfModels[t].w_0  for t in range(K) if t==t], dim = 1).sum(dim=1).view([-1,1]) 
                
#                 temp1 = torch.inverse(2*A_inv \
#                                         +2*alpha*(K)*torch.diag(torch.ones_like(self.ListOfModels[k].w_0.view(-1))) \
#                                         -2*A_inv@B@A_inv \
#                                         +2*A_inv@B@A_inv@B@A_inv)
#                 temp2 = 2*A_inv@self.ListOfModels[k].W \
#                         + 2*alpha*torch.cat([self.ListOfModels[t].w_0  for t in range(K) if t==t], dim = 1).sum(dim=1).view([-1,1]) \
#                         - (A_inv@B@A_inv@B+B@A_inv@B@A_inv)@second
                
    
                ListOfNewW0.append((temp1@temp2).detach())
            else:
                ListOfNewW0.append(None)
                
        for k in range(K):
            if self.ListOfModels[k].w_0 is not None:
                self.ListOfModels[k].w_0.data[:2, :] = ListOfNewW0[k].data[:2,:]
        return



class MixtureExpert:
    def __init__(self, input_dim = 10, K = 2, HyperParameters = {'beta': 1, 'alpha': 1}, HyperModel = None, ListOfModels = None, ListOfRegularizeModel = None, device = 'cpu'):
        """
        It's necessary! The Hyper Parameter should be additive to models.
        """
        
        self.K = K
        self.n = input_dim
        self.device = device
        
        self.HyperParameters = HyperParameters
        
        if HyperModel is None:
            self.HyperModel = HyperModelNN(input_dim = input_dim, hidden_dim = 10, output_dim = K, device = device)
        else:
            self.HyperModel = HyperModel
            
        if ListOfRegularizeModel is None:
            self.ListOfRegularizeModel = []
        else:
            self.ListOfRegularizeModel = ListOfRegularizeModel
            
        if ListOfModels is None:
            self.ListOfModels = [EachModelLinear(input_dim = input_dim, device = device) for _ in range(K)]
        else:
            self.ListOfModels = ListOfModels
        

        self.pZ = None
        return
        
    def E_step(self, X, Y):
# Optimize Z
        temp1 = self.HyperModel.LogPiExpectation(X, Y, self.HyperParameters)
        temp2 = torch.cat([self.ListOfModels[k].LogLikeLihoodExpectation(X, Y, self.HyperParameters) for k in range(self.K)], dim = 1)
        self.pZ = torch.nn.functional.softmax(temp1 + temp2, dim=-1).detach()
    
# Optimize each model
        for k in range(self.K):
            self.ListOfModels[k].E_step(X, Y, self.pZ[:,k].view([-1, 1]), self.HyperParameters)

# Do reqularization
        for k in range(len(self.ListOfRegularizeModel)):
            self.ListOfRegularizeModel[k].E_step(X, Y, self.pZ, self.HyperParameters)

# Optimize HyperModel
        self.HyperModel.E_step(X, Y, self.pZ, self.HyperParameters)
        return
        
    def M_step(self, X, Y):
# Optimize EachModel
        for k in range(self.K):
            self.ListOfModels[k].M_step(X, Y, self.pZ[:, k].view([-1, 1]), self.HyperParameters)
            
# Optimize HyperParameters
        for Parameter in self.HyperParameters:
            temp = None
            for k in range(self.K):
                ret = self.ListOfModels[k].OptimizeHyperParameters(X, Y, self.pZ[:, k].view([-1, 1]), self.HyperParameters, Parameter)
                if ret is not None:
                    if temp is None:
                        temp = 0
                    temp += ret
            
            if temp is not None:
                self.HyperParameters[Parameter] = temp.detach()
# Do reqularization
        for k in range(len(self.ListOfRegularizeModel)):
            self.ListOfRegularizeModel[k].M_step(X, Y, self.pZ, self.HyperParameters)

# # Optimize Z(testing)
#         temp1 = torch.nn.functional.log_softmax(self.HyperModel(X), dim = -1)
#         temp2 = torch.cat([self.ListOfModels[k].LogLikeLihoodExpectation(X, Y, self.HyperParameters) for k in range(self.K)], dim = 1)
#         pZ = torch.nn.functional.softmax(temp1 + temp2, dim=-1).detach()
# Optimize HyperModel
        self.HyperModel.M_step(X, Y, self.pZ, self.HyperParameters)
    


        return
                
    def fit(self, X = None, Y = None, epoch = 10, progress = None):
        """
        X has a shape [N x n]
        Y has a shape [n x p]
        """
        if X is None:
            return None
        if Y is None:
            return None
        
        iterations = range(epoch)
        
        if progress is not None:
            iterations = progress(iterations)
        
        for _ in iterations:
            self.E_step(X, Y)
            self.M_step(X, Y)
            
        return
    
    def predict(self, X):
        """
        X has a shape [N x n]
        """
        pi = torch.nn.functional.softmax(self.HyperModel(X), dim = -1).detach()
        answ = torch.cat([self.ListOfModels[k](X) for k in range(self.K)], dim = 1).detach()
        
        return (answ*pi).sum(dim = -1).data.numpy(), pi.data.numpy()



import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

class EachModel:
    def __init__(self):
        pass

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        raise NotImplementedError

    def OptimizeHyperParameters(self, X, Y, Z, HyperParameters, Parameter):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        Z is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        Parameter is a Key in dictionary
        """
        raise NotImplementedError

    def LogLikeLihoodExpectation(self, X, Y, HyperParameters):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        """
        raise NotImplementedError

    def E_step(self, X, Y, Z, HyperParameters):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        Z is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        """
        raise NotImplementedError

    def M_step(self, X, Y, Z, HyperParameters):
        """
        X is a tensor of shape [N x n]
        Y is a tensor of shape [N x 1]
        Z is a tensor of shape [N x 1]
        HyperParameters is a dictionary
        """
        raise NotImplementedError


class EachModelLinear(EachModel):
    def __init__(self, input_dim = 20, device = 'cpu', A = None, w = None):
        super(EachModelLinear, self).__init__()

        self.input_dim = input_dim
        self.device = device
        
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
            new_beta = ((temp1 + temp2 + temp3)*Z).mean()
            if new_beta > 0:
                return new_beta.detach()
            else:
                return (0*new_beta).detach()
        
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
            try:
                A_inv = torch.inverse(A)
            except:
                A_inv = (2**32)*torch.eye(A.shape[0])
            
            self.B = torch.inverse(A_inv + beta*((temp*Z.unsqueeze(1))@temp.transpose(2, 1)).sum(dim = 0)).detach()
            second = beta*(X*Y*Z).sum(dim = 0).view([-1, 1])       
            if self.w_0 is None:
                self.W.data = ((self.B@second)).view_as(self.W).detach()
            else:
                self.W.data = (self.B@(second + A_inv@self.w_0)).view_as(self.W).detach()
        
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
            self.w_0.data = self.W.data.clone()

        return
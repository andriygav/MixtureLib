import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset


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

                ListOfNewW0.append((temp1@temp2).detach())
            else:
                ListOfNewW0.append(None)
                
        for k in range(K):
            if self.ListOfModels[k].w_0 is not None:
                self.ListOfModels[k].w_0.data[:2, :] = ListOfNewW0[k].data[:2,:]
        return
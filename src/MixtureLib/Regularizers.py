import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset


class Regularizers:
    def __init__(self):
        pass

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
        Z is a tensor of shape [N x K]
        HyperParameters is a dictionary
        """
        raise NotImplementedError

class RegularizeModel(Regularizers):
    def __init__(self, ListOfModels = None, device = 'cpu'):
        super(RegularizeModel, self).__init__()

        if ListOfModels is None:
            self.ListOfModels = []
        else:
            self.ListOfModels = ListOfModels

        self.ListOfModelsW0 = []
        for k, LocalModel in enumerate(self.ListOfModels):
            if LocalModel.w_0 is not None:
                self.ListOfModelsW0.append((k, LocalModel.w_0.clone()))


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
        alpha = (1./(HyperParameters['alpha']+1e-30)).detach()

        beta = (1./(HyperParameters['beta'] + 0.000001)).detach()
        K = len(self.ListOfModels)
        
        ListOfNewW0 = []
        
        for k, w_0 in self.ListOfModelsW0:
            if len(self.ListOfModels[k].A.shape) == 1:
                try:
                    A_inv = torch.diag(1./self.ListOfModels[k].A)
                except:
                    A_inv = (2**32)*torch.ones(self.ListOfModels[k].A.shape[0])
            else:
                try:
                    A_inv = torch.inverse(self.ListOfModels[k].A)
                except:
                    A_inv = (2**32)*torch.eye(self.ListOfModels[k].A.shape[0])
            
            B = self.ListOfModels[k].B

            if len(alpha.shape) == 0:
                alpha = alpha*torch.diag(torch.ones_like(w_0.view(-1)))
            elif len(alpha.shape) == 1:
                alpha = torch.diag(alpha)

            temp1 = torch.inverse(A_inv + alpha*(K))
            temp2 = A_inv@self.ListOfModels[k].W \
                    + alpha@torch.cat([w_s_0 for t, w_s_0 in self.ListOfModelsW0 if t==t], dim = 1).sum(dim=1).view([-1,1]) 

            ListOfNewW0.append((k, (temp1@temp2).detach()))

        for (k, w_0), (t, new_w_0) in zip(self.ListOfModelsW0, ListOfNewW0):
            w_0.data = new_w_0.data

        for k, w_0 in self.ListOfModelsW0:
            if self.ListOfModels[k].w_0 is not None:
                self.ListOfModels[k].w_0.data = w_0.data.clone()

        return


class RegularizeFunc(Regularizers):
    def __init__(self, ListOfModels = None, R = lambda x: x.sum(), epoch=100, device = 'cpu'):
        super(RegularizeFunc, self).__init__()

        if ListOfModels is None:
            self.ListOfModels = []
        else:
            self.ListOfModels = ListOfModels

        self.ListOfModelsW0 = []
        for k, LocalModel in enumerate(self.ListOfModels):
            if LocalModel.w_0 is not None:
                self.ListOfModelsW0.append((k, LocalModel.w_0.clone()))

        self.epoch = epoch
                
        self.R = R


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
        
        W0_ = torch.tensor(torch.cat([w0[1] for w0 in self.ListOfModelsW0], dim = -1), requires_grad=True)
        W0 = W0_.transpose(0,1)

        optimizer = torch.optim.Adam([W0_])
        
        
        for i in range(self.epoch):
            loss = 0
            for local_model, w0  in zip(self.ListOfModels, W0):
                if local_model.A is not None:
                    if len(local_model.A.shape) == 1:
                        try:
                            A_inv = torch.diag(1./local_model.A)
                        except:
                            A_inv = (2**32)*torch.ones(local_model.A.shape[0])
                    else:
                        try:
                            A_inv = torch.inverse(local_model.A)
                        except:
                            A_inv = (2**32)*torch.eye(local_model.A.shape[0])


                    loss += -0.5*(w0@A_inv@w0)+0.5*w0@A_inv@local_model.W

            loss += self.R(W0)


            optimizer.zero_grad()
            (-loss).backward()
            optimizer.step()
            

        ListOfNewW0 = []
        
        for k, w_0 in enumerate(W0):
            ListOfNewW0.append((k, w_0.view([-1,1]).detach()))

        for (k, w_0), (t, new_w_0) in zip(self.ListOfModelsW0, ListOfNewW0):
            w_0.data = new_w_0.data

        for k, w_0 in self.ListOfModelsW0:
            if self.ListOfModels[k].w_0 is not None:
                self.ListOfModels[k].w_0.data = w_0.data.clone()

        return
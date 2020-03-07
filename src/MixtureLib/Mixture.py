import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

class Mixture:
    def __init__(self):
        pass

    def fit(self, X = None, Y = None, epoch = 10, progress = None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

class MixtureEM(Mixture):
    def __init__(self, input_dim = 10, K = 2, HyperParameters = {}, HyperModel = None, ListOfModels = None, ListOfRegularizeModel = None, device = 'cpu'):
        """
        It's necessary! The Hyper Parameter should be additive to models.
        """
        super(MixtureEM, self).__init__()
        self.K = K
        self.n = input_dim
        self.device = device
        
        self.HyperParameters = dict()
        for key in HyperParameters:
            self.HyperParameters[key] = torch.tensor(HyperParameters[key])
        
        if HyperModel is None:
            return None
            # self.HyperModel = HyperExpertNN(input_dim = input_dim, hidden_dim = 10, output_dim = K, device = device)
        else:
            self.HyperModel = HyperModel
            
        if ListOfRegularizeModel is None:
            self.ListOfRegularizeModel = []
        else:
            self.ListOfRegularizeModel = ListOfRegularizeModel
            
        if ListOfModels is None:
            return None
            # self.ListOfModels = [EachModelLinear(input_dim = input_dim, device = device) for _ in range(K)]
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
        pi = self.HyperModel.PredictPi(X, self.HyperParameters).detach()
        answ = torch.cat([self.ListOfModels[k](X) for k in range(self.K)], dim = 1).detach()
        
        return (answ*pi).sum(dim = -1).view([-1, 1]), pi.data.numpy()

class MixtureEMGatePrior(Mixture):
    def __init__(self, input_dim = 10, K = 2, lamb=0.5, HyperParameters = {}, HyperModel = None, ListOfModels = None, ListOfRegularizeModel = None, device = 'cpu'):
        """
        It's necessary! The Hyper Parameter should be additive to models.
        """
        super(MixtureEM, self).__init__()
        self.K = K
        self.n = input_dim
        self.device = device
        
        self.HyperParameters = dict()
        for key in HyperParameters:
            self.HyperParameters[key] = torch.tensor(HyperParameters[key])
        
        if HyperModel is None:
            return None
            # self.HyperModel = HyperExpertNN(input_dim = input_dim, hidden_dim = 10, output_dim = K, device = device)
        else:
            self.HyperModel = HyperModel
            
        if ListOfRegularizeModel is None:
            self.ListOfRegularizeModel = []
        else:
            self.ListOfRegularizeModel = ListOfRegularizeModel
            
        if ListOfModels is None:
            return None
            # self.ListOfModels = [EachModelLinear(input_dim = input_dim, device = device) for _ in range(K)]
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
        pi = self.HyperModel.PredictPi(X, self.HyperParameters).detach()
        answ = torch.cat([self.ListOfModels[k](X) for k in range(self.K)], dim = 1).detach()
        
        return (answ*pi).sum(dim = -1).view([-1, 1]), pi.data.numpy()

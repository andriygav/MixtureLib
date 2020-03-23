#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`mixturelib.regularizers` contains classes:

- :class:`mixturelib.mixture.Mixture`
- :class:`mixturelib.mixture.MixtureEmSample`
- :class:`mixturelib.mixture.MixtureEM`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

class Mixture:
    r"""Base class for all mixtures."""
    def __init__(self):
        r"""Constructor method
        """
        pass

    def fit(self, X = None, Y = None, epoch = 10, progress = None):
        r"""A method that fit a hyper model and local models in one procedure.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :param epoch: The number of epoch of training.
        :type epoch: int
        :param progress: The yield function for printing progress, like a tqdm.
            The function must take an iterator at the input and return 
            the same data.
        :type epoch: function
        """
        raise NotImplementedError

    def predict(self, X):
        r"""A method that predict value for given input data.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :return: The prediction of shape
            `num_elements` :math:`\times` `num_answers`.
        :rtype: FloatTensor
        """
        raise NotImplementedError


class MixtureEmSample(Mixture):
    r"""The implementation of EM-algorithm for solving the 
    two stage optimisation problem. 
    Unlike :class:`mixturelib.mixture.MixtureEM`, this class samples data 
    when fit local models.

    .. warning::
        All Hyper Parameters should be additive to models, when you wanna 
        optimize them.

    .. warning::
        It's necessary! The parameter `input_dim` will be depricated.

        It's necessary! The parameter `K` will be depricated.

    :param input_dim: The number of features.
    :type input_dim: int
    :param K: The number of local models.
    :type K: int
    :param HyperParameters: The dictionary of all hyper parametrs.
        Where `key` is string and `value` is float or FloatTensor.
    :param HyperModel: The hyper model which are weighted all local models.
    :type HyperModel: :class:`mixturelib.hyper_models.HyperModel`
    :param ListOfModels: The list of models with E_step and M_step methods.
    :type ListOfModels: list
    :param ListOfRegularizeModel: The list of regulizers with E_step and 
        M_step methods.
    :type ListOfRegularizeModel: list
    :param device: The device for pytorch. 
        Can be 'cpu' or 'gpu'. Default 'cpu'.
    :type device: string

    Example:

    >>> _ = torch.random.manual_seed(42) # Set random seed for repeatability
    >>>
    >>> w = torch.randn(2, 1) # Generate real parameter vector
    >>> X = torch.randn(12, 2) # Generate features data
    >>> Y = torch.cat(
    ...         [
    ...             X[:5]@first_w, 
    ...             X[5:10]@second_w, 
    ...             X[10:11]@first_w, 
    ...             X[11:]@second_w
    ...         ])
    ...     + 0.1 * torch.randn(12, 1) # Generate target data with noise 0.1
    >>>
    >>> first_model = EachModelLinear(
    ...     input_dim=2, 
    ...     A=torch.tensor([1., 1.]),
    ...     w=torch.tensor([[0.], [0.]])) # Init first local model
    >>> second_model = EachModelLinear(
    ...     input_dim=2,
    ...     A=torch.tensor([1., 1.]),
    ...     w=torch.tensor([[1.], [1.]])) # Init second local model
    >>> hyper_model = HyperExpertNN(
    ...     input_dim=2, 
    ...     output_dim=2) # Init hyper model with Diriclet weighting
    >>> hyper_parameters = {'beta': 1.} # Withor hyper parameters
    >>>
    >>> mixture = MixtureEmSample(
    ...     input_dim=2, K=2, HyperModel=hyper_model, 
    ...     HyperParameters=hyper_parameters, 
    ...     ListOfModels=[first_model, second_model]) # Init hyper model
    >>> mixture.fit(X[:10], Y[:10]) # Optimise model parameter
    >>>
    >>> mixture.predict(X[10:])[0].view(-1)
    tensor([-0.0129, -0.4518])
    >>> Y[10:].view(-1)
    tensor([-0.2571, -0.4907])
    """
    def __init__(self, input_dim = 10, K = 2, HyperParameters = {}, HyperModel = None, ListOfModels = None, ListOfRegularizeModel = None, device = 'cpu'):
        r"""Constructor method
        """
        super(MixtureEmSample, self).__init__()
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
        r"""Doing E-step of EM-algorigthm. This method call E_step for all 
        local models, for hyper model and for all regularizations step by step.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        """
# Optimize Z
        self.temp1 = temp1 = self.HyperModel.LogPiExpectation(X, Y, self.HyperParameters)
        self.temp2 = temp2 = torch.cat([self.ListOfModels[k].LogLikeLihoodExpectation(X, Y, self.HyperParameters) for k in range(self.K)], dim = 1)
        self.pZ = torch.nn.functional.softmax(temp1 + temp2, dim=-1).detach()
    
        self.temp2_proba = temp2_proba = torch.nn.functional.softmax(temp2, dim=-1).detach()
    
        self.indexes = torch.multinomial(self.pZ, num_samples=1).view(-1)
        prior_index = torch.multinomial(torch.nn.functional.softmax(torch.ones_like(self.pZ), dim=-1), num_samples=1).view(-1)
        
        self.lerning_indexes = []
        for k in range(self.K):
            ind_k = (self.indexes==k)
            ind_k *= (prior_index==k)
            
            if torch.sum(ind_k) < 3:
                ind_k = prior_index

            self.lerning_indexes.append(ind_k)
    
# Optimize each model
        for k in range(self.K):
            self.ListOfModels[k].E_step(X[self.lerning_indexes[k]], Y[self.lerning_indexes[k]], torch.ones_like(self.pZ[self.lerning_indexes[k], k]).view([-1, 1]), self.HyperParameters)

# Do reqularization
        for k in range(len(self.ListOfRegularizeModel)):
            self.ListOfRegularizeModel[k].E_step(X, Y, self.pZ, self.HyperParameters)

# Optimize HyperModel
        self.HyperModel.E_step(X, Y, temp2_proba+0*self.pZ, self.HyperParameters)
        return
        
    def M_step(self, X, Y):
        r"""Doing M-step of EM-algorigthm. This method call M_step for all 
        local models, for hyper model and for all regularizations step by step.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        """
# Optimize EachModel
        for k in range(self.K):
            self.ListOfModels[k].M_step(X[self.lerning_indexes[k]], Y[self.lerning_indexes[k]], torch.ones_like(self.pZ[self.lerning_indexes[k], k]).view([-1, 1]), self.HyperParameters)
            
# Optimize HyperParameters
        for Parameter in self.HyperParameters:
            temp = None
            for k in range(self.K):
                ret = self.ListOfModels[k].OptimizeHyperParameters(X[self.lerning_indexes[k]], Y[self.lerning_indexes[k]], torch.ones_like(self.pZ[self.lerning_indexes[k], k]).view([-1, 1]), self.HyperParameters, Parameter)
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
        r"""A method that fit a hyper model and local models in one procedure.

        Call E-step and M-step in each epoch.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :param epoch: The number of epoch of training.
        :type epoch: int
        :param progress: The yield function for printing progress, like a tqdm.
            The function must take an iterator at the input and return 
            the same data.
        :type epoch: function
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
        r"""A method that predict value and probability for each local models 
        for given input data.

        For each x from X predicts
        :math:`answer = \sum_{k=1}^{K}\pi_k\bigr(x\bigr)g_k\bigr(x\bigr)`, 
        where :math:`g_k` is a local model.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :return:
            The prediction of shape 
            `num_elements` :math:`\times` `num_answers`.
            
            The probability of shape
            `num_elements` :math:`\times` `num_models`.
        :rtype: FloatTensor, FloatTensor
        """
        pi = self.HyperModel.PredictPi(X, self.HyperParameters).detach()
        answ = torch.cat([self.ListOfModels[k](X) for k in range(self.K)], dim = 1).detach()
        
        return (answ*pi).sum(dim = -1).view([-1, 1]), pi.data.numpy()

class MixtureEM(Mixture):
    r"""The implementation of EM-algorithm for solving the 
    two stage optimisation problem. 
    Unlike :class:`mixturelib.mixture.MixtureEM`, this class uses analytical 
    solution when fit local models.

    .. warning::
        All Hyper Parameters should be additive to models, when you wanna 
        optimize them.

    .. warning::
        It's necessary! The parameter `input_dim` will be depricated.

        It's necessary! The parameter `K` will be depricated.

    :param input_dim: The number of features.
    :type input_dim: int
    :param K: The number of local models.
    :type K: int
    :param HyperParameters: The dictionary of all hyper parametrs.
        Where `key` is string and `value` is float or FloatTensor.
    :param HyperModel: The hyper model which are weighted all local models.
    :type HyperModel: :class:`mixturelib.hyper_models.HyperModel`
    :param ListOfModels: The list of models with E_step and M_step methods.
    :type ListOfModels: list
    :param ListOfRegularizeModel: The list of regulizers with E_step and 
        M_step methods.
    :type ListOfRegularizeModel: list
    :param device: The device for pytorch. 
        Can be 'cpu' or 'gpu'. Default 'cpu'.
    :type device: string

    Example:

    >>> _ = torch.random.manual_seed(42) # Set random seed for repeatability
    >>>
    >>> w = torch.randn(2, 1) # Generate real parameter vector
    >>> X = torch.randn(12, 2) # Generate features data
    >>> Y = torch.cat(
    ...         [
    ...             X[:5]@first_w, 
    ...             X[5:10]@second_w, 
    ...             X[10:11]@first_w, 
    ...             X[11:]@second_w
    ...         ])
    ...     + 0.1 * torch.randn(12, 1) # Generate target data with noise 0.1
    >>>
    >>> first_model = EachModelLinear(
    ...     input_dim=2, 
    ...     A=torch.tensor([1., 1.]),
    ...     w=torch.tensor([[0.], [0.]])) # Init first local model
    >>> second_model = EachModelLinear(
    ...     input_dim=2,
    ...     A=torch.tensor([1., 1.]),
    ...     w=torch.tensor([[1.], [1.]])) # Init second local model
    >>> hyper_model = HyperExpertNN(
    ...     input_dim=2, 
    ...     output_dim=2) # Init hyper model with Diriclet weighting
    >>> hyper_parameters = {'beta': 1.} # Withor hyper parameters
    >>>
    >>> mixture = MixtureEmSample(
    ...     input_dim=2, K=2, HyperModel=hyper_model, 
    ...     HyperParameters=hyper_parameters, 
    ...     ListOfModels=[first_model, second_model]) # Init hyper model
    >>> mixture.fit(X[:10], Y[:10]) # Optimise model parameter
    >>>
    >>> mixture.predict(X[10:])[0].view(-1)
    tensor([ 0.0878, -0.4212])
    >>> Y[10:].view(-1)
    tensor([-0.2571, -0.4907])
    """
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
        r"""Doing E-step of EM-algorigthm. This method call E_step for all 
        local models, for hyper model and for all regularizations step by step.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        """
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
        r"""Doing M-step of EM-algorigthm. This method call M_step for all 
        local models, for hyper model and for all regularizations step by step.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        """
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
        r"""A method that fit a hyper model and local models in one procedure.

        Call E-step and M-step in each epoch.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :param epoch: The number of epoch of training.
        :type epoch: int
        :param progress: The yield function for printing progress, like a tqdm.
            The function must take an iterator at the input and return 
            the same data.
        :type epoch: function
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
        r"""A method that predict value for given input data.

        For each x from X predicts
        :math:`answer = \sum_{k=1}^{K}\pi_k\bigr(x\bigr)g_k\bigr(x\bigr)`, 
        where :math:`g_k` is a local model.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :return:
            The prediction of shape 
            `num_elements` :math:`\times` `num_answers`.
            
            The probability of shape
            `num_elements` :math:`\times` `num_models`.
        :rtype: FloatTensor, FloatTensor
        """
        pi = self.HyperModel.PredictPi(X, self.HyperParameters).detach()
        answ = torch.cat([self.ListOfModels[k](X) for k in range(self.K)], dim = 1).detach()
        
        return (answ*pi).sum(dim = -1).view([-1, 1]), pi.data.numpy()
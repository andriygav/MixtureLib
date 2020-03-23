#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`mixturelib.hyper_models` contains classes:

- :class:`mixturelib.hyper_models.HyperModel`
- :class:`mixturelib.hyper_models.HyperModelDirichlet`
- :class:`mixturelib.hyper_models.HyperExpertNN`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset
class HyperModel:
    r"""Base class for all hyper models."""
    def __init__(self):
        """Constructor method
        """
        pass

    def E_step(self, X, Y, Z, HyperParameters):
        r"""Doing E-step of EM-algorithm. Finds variational probability `q` 
        of model parameters.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape 
            `num_elements` :math:`\times` `num_models`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        raise NotImplementedError

    def M_step(self, X, Y, Z, HyperParameters):
        r"""Doing M-step of EM-algorithm. Finds model hyper parameters.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape 
            `num_elements` :math:`\times` `num_models`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        raise NotImplementedError

    def LogPiExpectation(self, X, Y, HyperParameters):
        r"""Returns the expected value of each models probability.
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        raise NotImplementedError

    def PredictPi(self, X, HyperParameters):
        r"""Returns the probability of each models.
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        raise NotImplementedError   


class HyperModelDirichlet(HyperModel):
    r"""A hyper model for mixture of model. The hyper model cannot predict 
    local model for each object, because model probability does not 
    depend on object.

    In this hyper model, the probability of each local model is a vector 
    from dirichlet distribution.

    :param output_dim: The number of local models.
    :type output_dim: int
    :param device: The device for pytorch. 
        Can be 'cpu' or 'gpu'. Default 'cpu'.

    Example:

    >>> _ = torch.random.manual_seed(42) # Set random seed for repeatability
    >>>
    >>> w = torch.randn(2, 1) # Generate real parameter vector
    >>> X = torch.randn(5, 2) # Generate features data
    >>> Z = torch.distributions.dirichlet.Dirichlet(
    ...     torch.tensor([0.5, 0.5])).sample(
    ...         (5,)) # Set corresponding between data and local models.
    >>> Y = X@w + 0.1*torch.randn(5, 1) # Generate target data with noise 0.1
    >>>
    >>> hyper_model = HyperModelDirichlet(
    ...     output_dim=2) # Init hyper model with Diriclet weighting
    >>> hyper_parameters = {} # Withor hyper parameters
    >>>
    >>> hyper_model.LogPiExpectation(
    ...     X, Y, hyper_parameters) # Log of probability before E step
    tensor([[-1.0000, -1.0000],
            [-1.0000, -1.0000],
            [-1.0000, -1.0000],
            [-1.0000, -1.0000],
            [-1.0000, -1.0000]])
    >>> 
    >>> hyper_model.E_step(X, Y, Z, hyper_parameters)
    >>> hyper_model.LogPiExpectation(
    ...     X, Y, hyper_parameters)  # Log of probability after E step
    tensor([[-0.7118, -0.8310],
            [-0.7118, -0.8310],
            [-0.7118, -0.8310],
            [-0.7118, -0.8310],
            [-0.7118, -0.8310]])
    """
    def __init__(self, output_dim = 2, device = 'cpu'):
        """Constructor method
        """
        super(HyperModelDirichlet, self).__init__()
        self.output_dim = output_dim
        self.device = device
        
        self.mu = torch.ones(self.output_dim)
        self.m = torch.zeros_like(self.mu)
        self.m.data = self.mu.data.clone()
        self.N = 0
    
    def E_step(self, X, Y, Z, HyperParameters):
        r"""Doing E-step of EM-algorithm. Finds variational probability `q` 
        of model parameters.

        Calculate analytical solution for estimate `q` in the class of 
        normal distributions :math:`q = Dir(m)`, where
        :math:`m = \mu + \gamma`, where 
        :math:`\gamma_k = \sum_{i=1}^{num\_elements}Z_{ik}`, and 
        :math:`\mu` is prior.

        .. warning::
            Now :math:`\mu_k` is `1` for all `k`, and can not be changed.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape 
            `num_elements` :math:`\times` `num_models`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        gamma = Z.sum(dim=0)
        self.m = (self.mu + gamma).detach()
        self.N = Z.shape[0]
        pass
    
    def M_step(self, X, Y, Z, HyperParameters):
        r"""The method does nothing.
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape 
            `num_elements` :math:`\times` `num_models`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        pass

    def LogPiExpectation(self, X, Y, HyperParameters):
        r"""Returns the expected value of each models log of probability.

        Returns the expectation of :math:`\log \pi` value where 
        :math:`\pi` is a random value from Dirichlet distribution.

        This function calculates by using :math:`\digamma` function
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict

        :return: The tensor of shape 
            `num_elements` :math:`\times` `num_models`. The espected value of 
            each models probability.
        :rtype: FloatTensor
        """
        temp_1 = torch.ones([X.shape[0], self.output_dim])
        temp_2 = (torch.digamma(self.m) - torch.digamma(self.output_dim*self.mu + self.N))
        return temp_1*temp_2

    def PredictPi(self, X, HyperParameters):
        r"""Returns the probability (weight) of each models.

        Return the same vector :math:`\pi` for all object. 
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict

        :return: The tensor of shape 
            `num_elements` :math:`\times` `num_models`. 
            The probability (weight) of each models.
        :rtype: FloatTensor
        """
        temp_1 = torch.ones([X.shape[0], self.output_dim])
        temp_2 = (torch.digamma(self.m) - torch.digamma(self.output_dim*self.mu + self.N))
        return temp_1*temp_2


class HyperExpertNN(nn.Module, HyperModel):
    r"""A hyper model for mixture of experts. The hyper model prediction on 
    local models probability are depend on the object.

    In this hyper model, the probability of each local model is a 
    neural network prediction with softmax. Neural network is a three layer 
    fully conected neural network.

    :param input_dim: The number of features.
    :type input_dim: int
    :param hidden_dim: The number of parameters in hidden layer.
    :type hidden_dim: int
    :param output_dim: The number of local models.
    :type output_dim: int
    :param epochs: The number epoch to train neural network in each step.
    :type epochs: int
    :param device: The device for pytorch. 
        Can be 'cpu' or 'gpu'. Default 'cpu'.

    Example:

    >>> _ = torch.random.manual_seed(42) # Set random seed for repeatability
    >>>
    >>> w = torch.randn(2, 1) # Generate real parameter vector
    >>> X = torch.randn(5, 2) # Generate features data
    >>> Z = torch.distributions.dirichlet.Dirichlet(
    ...     torch.tensor([0.5, 0.5])).sample(
    ...         (5,)) # Set corresponding between data and local models.
    >>> Y = X@w + 0.1*torch.randn(5, 1) # Generate target data with noise 0.1
    >>>
    >>> hyper_model = HyperExpertNN(
    ...     input_dim=2, 
    ...     output_dim=2) # Init hyper model with Diriclet weighting
    >>> hyper_parameters = {} # Withor hyper parameters
    >>>
    >>> hyper_model.LogPiExpectation(
    ...     X, Y, hyper_parameters) # Log of probability before E step
    tensor([[-0.4981, -0.9356],
            [-0.5176, -0.9063],
            [-0.4925, -0.9443],
            [-0.4957, -0.9395],
            [-0.4969, -0.9376]])
    >>> 
    >>> hyper_model.E_step(X, Y, Z, hyper_parameters)
    >>> hyper_model.LogPiExpectation(
    ...     X, Y, hyper_parameters)  # Log of probability after E step
    tensor([[-0.6294, -0.7612],
            [-0.9327, -0.5000],
            [-0.3273, -1.2760],
            [-0.5775, -0.8239],
            [-0.5357, -0.8801]])
    """
    def __init__(self, input_dim = 20, hidden_dim = 10, output_dim = 10, epochs=100, device = 'cpu'):
        """Constructor method
        """
        super(HyperExpertNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.epochs=epochs
        
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters())
        
        self.to(device)
        
    def forward(self, input):
        r"""Returns model prediction for the given input data. 

        .. warning::
            The number `num_answers` can be just `1`.
        
        :param input: The tensor of shape
            `num_elements` :math:`\times` `num_feature`.
        :type input: FloatTensor.

        :return: The tensor of shape
            `num_elements` :math:`\times` `num_models`.
            Model prediction of probability for all local models for the 
            given input data.
        :rtype: FloatTensor
        """
        out = input
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out
    
    def E_step(self, X, Y, Z, HyperParameters):
        r"""The method does nothing.
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape `num_elements` :math:`\times` `num_models`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        pass
    
    def M_step(self, X, Y, Z, HyperParameters):
        r"""Doing M-step of EM-algorithm. Finds model parameters by using 
        gradient descent.

        Parameters are optimized with respect to the loss function
        :math:`loss = -\sum_{i=1}^{num\_elements}\sum_{k=1}^{num\_models}
        \log\pi_k(x_i, V)`, where `V` is a neural network parameters. 
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape 
            `num_elements` :math:`\times` `num_models`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
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
        r"""Returns the expected value of each models log of probability.

        Takes log softmax from the forward method.
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict

        :return: The tensor of shape 
            `num_elements` :math:`\times` `num_models`. The espected value of 
            each models probability.
        :rtype: FloatTensor
        """
        return torch.nn.functional.log_softmax(self.forward(X), dim = -1)
    
    def PredictPi(self, X, HyperParameters):
        r"""Returns the probability (weight) of each models.

        Takes softmax from the forward method.
        
        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict

        :return: The tensor of shape 
            `num_elements` :math:`\times` `num_models`. 
            The probability (weight) of each models.
        :rtype: FloatTensor
        """
        return torch.nn.functional.softmax(self.forward(X), dim = -1)


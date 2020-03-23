#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`mixturelib.local_models` contains classes:

- :class:`mixturelib.local_models.EachModel`
- :class:`mixturelib.local_models.EachModelLinear`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

class EachModel:
    r"""Base class for all local models."""
    def __init__(self):
        """Constructor method
        """
        pass

    def __call__(self, input):
        r"""Returns model prediction for the given input data.
        
        :param input: The tensor of shape
            `num_elements` :math:`\times` `num_feature`.
        :type input: FloatTensor.
        :return: The tensor of shape
            `num_elements` :math:`\times` `num_answers`.
            Model answers for the given input data
        :rtype: FloatTensor
        """
        return self.forward(input)

    def forward(self, input):
        r"""Returns model prediction for the given input data.
        
        :param input: The tensor of shape
            `num_elements` :math:`\times` `num_feature`.
        :type input: FloatTensor.

        :return: The tensor of shape
            `num_elements` :math:`\times` `num_answers`.
            Model answers for the given input data
        :rtype: FloatTensor
        """
        raise NotImplementedError

    def OptimizeHyperParameters(self, X, Y, Z, HyperParameters, Parameter):
        r"""Returns the local part of new Parameter.

        .. warning::
            Returned local part must be aditive to Parameter, because new 
            value of Parameter is sum of local part from all local model.

        .. warning::
            The number `num_answers` can be just `1`.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape `num_elements` :math:`\times` `1`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        :param Parameter: The key from `HyperParameters` dictionary.
            The name of the hyperparameter to be optimized.
        :type Parameter: string

        :return: A local part of new `HyperParameters[Parameter]` value.
        :rtype: FloatTensor       
        """
        raise NotImplementedError

    def LogLikeLihoodExpectation(self, X, Y, HyperParameters):
        r"""Returns expected log-likelihod of a given vector of answers for
        given data seperated. The expectation is taken according to the model 
        parameters :math:`\mathsf{E}p\bigr(Y|X\bigr)`.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict

        :return: The tensor of shape `num_elements` :math:`\times` `1`.
        :rtype: FloatTensor
        """
        raise NotImplementedError

    def E_step(self, X, Y, Z, HyperParameters):
        r"""Doing E-step of EM-algorithm. Finds variational probability `q` 
        of model parameters.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape `num_elements` :math:`\times` `1`.
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
        :param Z: The tensor of shape `num_elements` :math:`\times` `1`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        raise NotImplementedError


class EachModelLinear(EachModel):
    r"""A model for solving the linear regression problem
    :math:`\textbf{y} = \textbf{x}^{\mathsf{T}}\textbf{w}`. The model uses an analytical
    solution for estimation :math:`\textbf{w}`. Also model finds a distribution
    of model parameters :math:`\textbf{w}`.

    Distribution is 
    :math:`\textbf{w} \sim \mathcal{N}\bigr(\hat{\textbf{w}}, \textbf{A}\bigr)`

    .. warning::
        The priors `A` and `w` must be set or not together.

    :param input_dim: The number of feature in each object. Must be positive.
    :type input_dim: int
    :param device: The device for pytorch. 
        Can be 'cpu' or 'gpu'. Default 'cpu'.
    :type device: string
    :param A: The tensor of shape input_dim :math:`\times` input_dim. It is a 
        prior covariance matrix for model parameters.
        Also can be the tensor of shape input_dim. In this case, it is a 
        diagonal of prior covariance matrix for model parameters, and all 
        nondiagonal values are zerous.
    :type A: FloatTensor
    :param w: The tensor of shape input_dim. It is a prior mean for model 
        parameters
    :type w: FloatTensor
    :param OptimizedHyper: The set of hyperparameters that will be optimized.
        Default all hyperparameters will be optimized.
    :type OptimizedHyper: set

    Example:

    >>> _ = torch.random.manual_seed(42) # Set random seed for repeatability
    >>>
    >>> w = torch.randn(2, 1) # Generate real parameter vector
    >>> X = torch.randn(10, 2) # Generate features data
    >>> Z = torch.ones(10, 1) # Set that all data correspond to this model
    >>> Y = X@w + 0.1*torch.randn(10, 1) # Generate target data with noise 0.1
    >>>
    >>> linear_model = EachModelLinear(
    ...     input_dim=2) # Init linear model withot any priors
    >>> hyper_parameters = {
    ...     'beta': torch.tensor(1.)} # Init noise level beta
    >>>
    >>> linear_model.E_step(
    ...     X[:8], Y[:8], Z[:8], hyper_parameters) # Optimise model parameter
    >>>
    >>> linear_model(X[8:]).view(-1) # Model prediction for the test part
    tensor([-0.1975, -0.2427])
    >>> Y[8:].view(-1) # Real target for test part
    tensor([-0.2124, -0.1837])
    """
    def __init__(self, input_dim = 20, device = 'cpu', A = None, w = None, OptimizedHyper=set(['w_0', 'A', 'beta'])):
        """Constructor method
        """
        super(EachModelLinear, self).__init__()

        self.input_dim = input_dim
        self.device = device
        self.OptimizedHyper = OptimizedHyper
        self.A = A
            
        self.W = (1e-5)*torch.randn(input_dim, 1, device = self.device)
        r"""Object :math:`\textbf{W}` is a model parameters"""

        if w is not None:
            self.w_0 = w.clone()
            self.W.data = w.data.clone() + (1e-5)*torch.randn(input_dim, 1, device = self.device)
        else:
            self.w_0 = w
        
        self.B = torch.eye(input_dim, device = self.device)
        r"""Object :math:`\textbf{B}` is a covariance matrix for variational 
        distribution"""

        if self.A is not None:
            if len(self.A.shape) == 1:
                self.B.data = torch.diag(self.A).data.clone()
            else:
                self.B.data = self.A.data.clone()
        
    def forward(self, input):
        r"""Returns model prediction for the given input data.
        Linear prediction is :math:`input@w`.

        .. warning::
            The number `num_answers` can be just `1`.
        
        :param input: The tensor of shape
            `num_elements` :math:`\times` `num_feature`.
        :type input: FloatTensor.

        :return: The tensor of shape
            `num_elements` :math:`\times` `num_answers`.
            Model answers for the given input data
        :rtype: FloatTensor
        """
        return input@self.W
    
    def OptimizeHyperParameters(self, X, Y, Z, HyperParameters, Parameter):
        r"""Returns the local part of new Parameter.
    
        In this case `local_part` is inverse beta:
        :math:`\frac{1}{\beta} = \frac{1}{num\_elements}
        \sum_{i=1}^{num\_elements}[Y_i^2-2Y_iX_i^{\mathsf{T}}\hat{\textbf{w}} +
        X_i^{\mathsf{T}}\mathsf{E}[ww^{\mathsf{T}}]X_i]`

        .. warning::
            Return local part must be aditive to Parameter, because new 
            value of Parameter is sum of local part from all local model.

        .. warning::
            HyperParameters must contain `beta` hyperparameter.

        .. warning::
            The number `num_answers` can be just `1`.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape `num_elements` :math:`\times` `1`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        :param Parameter: The key from `HyperParameters` dictionary.
            The name of the hyperparameter to be optimized.
        :type Parameter: string

        :return: A local part of new `HyperParameters[Parameter]` value.
        :rtype: FloatTensor
        """
        if 'beta' in self.OptimizedHyper:
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
        r"""Returns expected log-likelihod of a given vector of answers for
        given data seperated. The expectation is taken according to the model 
        parameters :math:`\mathsf{E}p\bigr(Y|X\bigr)`.

        .. warning::
            HyperParameters must contain `beta` hyperparameter.

        .. warning::
            The number `num_answers` can be just `1`.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict

        :return: The tensor of shape `num_elements` :math:`\times` `1`.
        :rtype: FloatTensor
        """
        beta = 1./(HyperParameters['beta'] + 0.000001)
        temp1 = Y**2
        temp2 = -2*Y*(X@self.W)
        temp3 = torch.diagonal(X@(self.B+self.W@self.W.transpose(0,1))@X.transpose(0,1)).view([-1, 1])
        return (-0.5*beta*(temp1 + temp2 + temp3) + 0.5*math.log(beta/(2*math.pi))).detach()
        

    def E_step(self, X, Y, Z, HyperParameters):
        r"""Doing E-step of EM-algorithm. Finds variational probability `q` 
        of model parameters.

        Calculate analytical solution for estimate `q` in the class of 
        normal distributions :math:`q = \mathcal{N}\bigr(m, B\bigr)`, where
        :math:`B = (A^{-1}+\beta\sum_{i=1}X_iX_i^{\mathsf{T}}\mathsf{E}z_{i})`
        and :math:`m = B(A^{-1}w_0+\beta\sum_{i=1}X_iY_i\mathsf{E}z_{i})`

        .. warning::
            HyperParameters must contain `beta` hyperparameter.

        .. warning::
            The number `num_answers` can be just `1`.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape `num_elements` :math:`\times` `1`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
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
        r"""Doing M-step of EM-algorithm. Finds model hyper parameters.

        .. warning::
            HyperParameters must contain `beta` hyperparameter.

        .. warning::
            The number `num_answers` can be just `1`.

        :param X: The tensor of shape 
            `num_elements` :math:`\times` `num_feature`.
        :type X: FloatTensor
        :param Y: The tensor of shape 
            `num_elements` :math:`\times` `num_answers`.
        :type Y: FloatTensor
        :param Z: The tensor of shape `num_elements` :math:`\times` `1`.
        :type Z: FloatTensor
        :param HyperParameters: The dictionary of all hyper parametrs.
            Where `key` is string and `value` is FloatTensor.
        :type HyperParameters: dict
        """
        beta = 1./(HyperParameters['beta'] + 0.000001)

        if 'A' in self.OptimizedHyper:
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
                
        if 'w_0' in self.OptimizedHyper:
            if self.w_0 is not None:
                self.w_0.data = self.W.data.clone()

        return
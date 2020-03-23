#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`mixturelib.regularizers` contains classes:

- :class:`mixturelib.regularizers.Regularizers`
- :class:`mixturelib.regularizers.RegularizeModel`
- :class:`mixturelib.regularizers.RegularizeFunc`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

class Regularizers:
    r"""Base class for all regulizers."""
    def __init__(self):
        """Constructor method
        """
        pass

    def E_step(self, X, Y, Z, HyperParameters):
        r"""Make some regularization on the E-step.
        
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
        r"""Make some regularization on the M-step.

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

class RegularizeModel(Regularizers):
    r"""The class of regularization to create a relationship between
    prior means. The relationship between the parameters in this case, is that 
    the mean distributions should be equal.

    .. warning::
        All local models must be Linear model for the regression task.
        Also can be used :class:`mixturelib.local_models.EachModelLinear`.

    This Regularizer make correction on the M-step for each Linear Model.

    :param ListOfModels: A list of local models to be regularized.
    :type ListOfModels: list
    :param device: The device for pytorch. 
        Can be 'cpu' or 'gpu'. Default 'cpu'.
    :type device: string

    Example:

    >>> _ = torch.random.manual_seed(42) # Set random seed for repeatability
    >>>
    >>> w = torch.randn(2, 1) # Generate real parameter vector
    >>> X = torch.randn(10, 2) # Generate features data
    >>> Z = torch.ones(10, 1) # Set that all data correspond to this model
    >>> Y = X@w + 0.1*torch.randn(10, 1) # Generate target data with noise 0.1
    >>>
    >>> first_model = EachModelLinear(
    ...     input_dim=2, 
    ...     A=torch.tensor([1., 1.]),
    ...     w=torch.tensor([[0.], [0.]])) # Init first local model
    >>> second_model = EachModelLinear(
    ...     input_dim=2,
    ...     A=torch.tensor([1., 1.]),
    ...     w=torch.tensor([[1.], [1.]])) # Init second local model
    >>> hyper_parameters = {
    ...     'alpha': torch.tensor([1., 1e-10])} # Set regularization parameter
    >>>
    >>> first_model.w_0, first_model.W # First prior and paramaters before
    (tensor([[0.],
             [0.]]),
    tensor([[1.3314e-06],
            [8.6398e-06]]))
    >>> second_model.w_0, second_model.W # Second prior and paramaters before
    (tensor([[1.],
             [1.]]),
     tensor([[1.0000],
             [1.0000]]))
    >>>
    >>> Rg = RegularizeModel(
    ...     ListOfModels=[first_model, second_model]) # Set regulariser
    >>> _ = Rg.M_step(X, Y, Z, hyper_parameters) # Regularize
    >>>
    >>> first_model.w_0, first_model.W # First prior and paramaters after
    (tensor([[0.3333],
             [0.5000]]),
     tensor([[1.3314e-06],
             [8.6398e-06]]))
    >>> second_model.w_0, second_model.W # Second prior and paramaters after
    (tensor([[0.6667],
             [0.5000]]),
     tensor([[1.0000],
             [1.0000]]))
    """

    def __init__(self, ListOfModels = None, device = 'cpu'):
        """Constructor method
        """
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
        r"""Make some regularization on the M-step.

        For all local model from ListOfModels with prior, make next 
        regularization :math:`w^0_k = \left[A_k^{-1} + (num\_models-1)\alpha\right]
        \left(A_k^{-1}\mathsf{E}w_k + \alpha\sum_{k'\not=k}w_k'\right)`

        .. warning::
            HyperParameters must contain `alpha` hyperparameter.

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
        alpha = (1./(HyperParameters['alpha']+1e-30)).detach()

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
    r"""The class of regularization to create any relationship between
    prior means. The relationship between the parameters is set by using 
    the link function.

    In the M-step solves next optimisation problem
    :math:`\sum_{k=1}^{num\_models}\left[-\frac{1}{2}w_k^0A_k^{-1}w_k^0+
    w_k^0A_k^{-1}\mathsf{E}w_k\right] + R(W^0) \to \infty`.

    .. warning::
        All local models must be Linear model for the regression task.
        Also can be used :class:`mixturelib.local_models.EachModelLinear`.

    .. warning::
        Link function represent a likelihood. This function will be 
        maximizing during optimisation.

    This Regularizer make correction on the M-step for each Linear Model.

    :param ListOfModels: A list of local models to be regularized.
    :type ListOfModels: list
    :param device: The device for pytorch. 
        Can be 'cpu' or 'gpu'. Default 'cpu'.
    :type device: string
    :param R: The link function between prior means for all local models. 
        The function must be scalar with type FloatTensor.
    :type R: function
    :param epoch: The number of epoch for solving optimisation problem in
        the M-step.
    :type epoch: int

    Example:

    >>> _ = torch.random.manual_seed(42) # Set random seed for repeatability
    >>>
    >>> w = torch.randn(2, 1) # Generate real parameter vector
    >>> X = torch.randn(10, 2) # Generate features data
    >>> Z = torch.ones(10, 1) # Set that all data correspond to this model
    >>> Y = X@w + 0.1*torch.randn(10, 1) # Generate target data with noise 0.1
    >>>
    >>> first_model = EachModelLinear(
    ...     input_dim=2, 
    ...     A=torch.tensor([1., 1.]),
    ...     w=torch.tensor([[0.], [0.]])) # Init first local model
    >>> second_model = EachModelLinear(
    ...     input_dim=2,
    ...     A=torch.tensor([1., 1.]),
    ...     w=torch.tensor([[1.], [1.]])) # Init second local model
    >>> hyper_parameters = {
    ...     'alpha': torch.tensor([1., 1e-10])} # Set regularization parameter
    >>>
    >>> first_model.w_0, first_model.W # First prior and paramaters before
    (tensor([[0.],
             [0.]]),
    tensor([[1.3314e-06],
            [8.6398e-06]]))
    >>> second_model.w_0, second_model.W # Second prior and paramaters before
    (tensor([[1.],
             [1.]]),
     tensor([[1.0000],
             [1.0000]]))
    >>>
    >>> Rg = RegularizeModel(
    ...     ListOfModels=[first_model, second_model],
    ...     R = lambda x: -(x**2).sum()) # Set regulariser
    >>> _ = Rg.M_step(X, Y, Z, hyper_parameters) # Regularize
    >>>
    >>> first_model.w_0, first_model.W # First prior and paramaters after
    (tensor([[4.8521e-06],
             [6.7789e-06]]),
     tensor([[1.3314e-06],
             [8.6398e-06]]))
    >>> second_model.w_0, second_model.W # Second prior and paramaters after
    (tensor([[0.9021],
             [0.9021]]),
     tensor([[1.0000],
             [1.0000]]))
    """

    def __init__(self, ListOfModels = None, R = lambda x: x.sum(), epoch=100, device = 'cpu'):
        """Constructor method
        """
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
        r"""Make some regularization on the M-step.

        Solves next optimisation problem
        :math:`\sum_{k=1}^{num\_models}\left[-\frac{1}{2}w_k^0A_k^{-1}w_k^0+
        w_k^0A_k^{-1}\mathsf{E}w_k\right] + R(W^0) \to \infty`.

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
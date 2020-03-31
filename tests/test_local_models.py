#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch

from mixturelib.local_models import EachModelLinear, EachModel

def test_EachModelLinear_init():
    model = EachModelLinear(input_dim = 20, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    assert model.input_dim == 20
    assert model.device == 'cpu'
    assert model.A is None
    assert model.w_0 is None
    assert model.W is not None
    assert model.OptimizedHyper == set(['w_0', 'A', 'beta'])

    assert model.B.shape == (20, 20)

def test_EachModel():
    model = EachModel()

    with pytest.raises(NotImplementedError):
        model.forward(None)

    with pytest.raises(NotImplementedError):
        model.OptimizeHyperParameters(None, None, None, None, None)

    with pytest.raises(NotImplementedError):
        model.LogLikeLihoodExpectation(None, None, None)

    with pytest.raises(NotImplementedError):
        model.E_step(None, None, None, None)

    with pytest.raises(NotImplementedError):
        model.M_step(None, None, None, None)



def test_EachModelLinear_forward():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].item() == 1.0861027703867876e-06
    assert answer[1][0].item() == -4.020557753392495e-06

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].item() == 1.0861027703867876e-06
    assert answer[1][0].item() == -4.020557753392495e-06

def test_EachModelLinear_OptimizeHyperParameters():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    answer = model.OptimizeHyperParameters(X, Y, Z, HyperParameters, 'beta')
    assert round(answer.item(), 2) == 1.38

    Z = torch.zeros(2, 1)
    answer = model.OptimizeHyperParameters(X, Y, Z, HyperParameters, 'beta')
    assert answer.item() == 0.

def test_EachModelLinear_LogLikeLihoodExpectation():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    answer = model.LogLikeLihoodExpectation(X, Y, HyperParameters)
    assert answer.shape == (2, 1)
    assert answer[0][0].item() == -3.4110240936279297
    assert answer[1][0].item() == -1.7702181339263916

def test_EachModelLinear_E_step_non_w_non_A():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    model.E_step(X, Y, Z, HyperParameters)

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 2
    assert answer[1][0].long().item() == 0

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 2
    assert answer[1][0].long().item() == 0

    assert model.B.shape == (2, 2)

    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.zeros(2,2), w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))
    model.E_step(X, Y, Z, HyperParameters)

def test_EachModelLinear_E_step_non_w_vec_A():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    model.E_step(X, Y, Z, HyperParameters)

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

    assert model.B.shape == (2, 2)

def test_EachModelLinear_E_step_non_w_mat_A():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.eye(2), 
                            w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    model.E_step(X, Y, Z, HyperParameters)

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

    assert model.B.shape == (2, 2)

def test_EachModelLinear_E_step_ver_w_mat_A():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.eye(2), 
                            w = torch.tensor([1., 1.]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    model.E_step(X, Y, Z, HyperParameters)

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

    assert model.B.shape == (2, 2)

def test_EachModelLinear_M_step_non_w_non_A():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))
    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    model.M_step(X, Y, Z, HyperParameters)

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].item() == 1.0861027703867876e-06
    assert answer[1][0].item() == -4.020557753392495e-06

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].item() == 1.0861027703867876e-06
    assert answer[1][0].item() == -4.020557753392495e-06

    assert model.A is None
    assert model.w_0 is None

def test_EachModelLinear_M_step_non_w_vec_A():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    model.M_step(X, Y, Z, HyperParameters)

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

def test_EachModelLinear_M_step_non_w_mat_A():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.eye(2), 
                            w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    model.M_step(X, Y, Z, HyperParameters)

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == 0
    assert answer[1][0].long().item() == 0

def test_EachModelLinear_M_step_ver_w_mat_A():
    torch.manual_seed(42)
    model = EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.eye(2), 
                            w = torch.tensor([1., 1.]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta']))

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    model.M_step(X, Y, Z, HyperParameters)

    answer = model(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == -1
    assert answer[1][0].long().item() == 1

    answer = model.forward(X)
    assert answer.shape == (2, 1)
    assert answer[0][0].long().item() == -1
    assert answer[1][0].long().item() == 1

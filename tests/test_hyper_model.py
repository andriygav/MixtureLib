#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch

from mixturelib.hyper_models import HyperModel
from mixturelib.hyper_models import HyperModelDirichlet
from mixturelib.hyper_models import HyperModelGateSparsed
from mixturelib.hyper_models import HyperExpertNN

def test_HyperModel():
    model = HyperModel()

    with pytest.raises(NotImplementedError):
        model.E_step(None, None, None, None)

    with pytest.raises(NotImplementedError):
        model.M_step(None, None, None, None)

    with pytest.raises(NotImplementedError):
        model.LogPiExpectation(None, None, None)

    with pytest.raises(NotImplementedError):
        model.PredictPi(None, None)

def test_HyperModelDirichlet_init():
    torch.manual_seed(42)

    hyper_model = HyperModelDirichlet(output_dim = 2, device = 'cpu')

    assert hyper_model.output_dim == 2
    assert hyper_model.device == 'cpu'
    assert (hyper_model.mu == torch.ones(2)).all()
    assert (hyper_model.m == hyper_model.mu).all()
    assert hyper_model.N == 0


def test_HyperModelDirichlet_E_step():
    torch.manual_seed(42)

    hyper_model = HyperModelDirichlet(output_dim = 2, device = 'cpu')

    
    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    hyper_model.E_step(X, Y, Z, HyperParameters)

    assert hyper_model.output_dim == 2
    assert hyper_model.device == 'cpu'
    assert (hyper_model.mu == torch.ones(2)).all()
    assert (hyper_model.m.long() == torch.tensor([3, 0])).all()
    assert hyper_model.N == 2

def test_HyperModelDirichlet_M_step():
    torch.manual_seed(42)

    hyper_model = HyperModelDirichlet(output_dim = 2, device = 'cpu')

    
    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    hyper_model.M_step(X, Y, Z, HyperParameters)

    assert hyper_model.output_dim == 2
    assert hyper_model.device == 'cpu'
    assert (hyper_model.mu == torch.ones(2)).all()
    assert (hyper_model.m == hyper_model.mu).all()
    assert hyper_model.N == 0

def test_HyperModelDirichlet_LogPiExpectation():
    torch.manual_seed(42)

    hyper_model = HyperModelDirichlet(output_dim = 2, device = 'cpu')

    
    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    log_pi = hyper_model.LogPiExpectation(X, Y, HyperParameters)

    assert log_pi.sum().long().item() == -4

def test_HyperModelDirichlet_PredictPi():
    torch.manual_seed(42)

    hyper_model = HyperModelDirichlet(output_dim = 2, device = 'cpu')

    X = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    pi = hyper_model.PredictPi(X, HyperParameters)

    assert pi.sum().long().item() == 2

    hyper_model = HyperModelDirichlet(output_dim = 2, device = 'cpu')

    X = torch.randn(2, 2)
    Z = -1*torch.ones(1, 2)
    HyperParameters = {'beta': torch.tensor(1.)}
    hyper_model.E_step(X, None, Z, HyperParameters)

    assert (hyper_model.m == torch.tensor([0., 0.])).all()
    
    pi = hyper_model.PredictPi(X, HyperParameters)
    assert (pi == 0.).all()



def test_HyperExpertNN_init():
    torch.manual_seed(42)

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=10,
                                device = 'cpu')

    params = hyper_model.parameters()
    sums = 0.
    for param in params:
        sums += param.sum()

    assert sums.long() == 4
    assert hyper_model.input_dim == 2
    assert hyper_model.output_dim == 2
    assert hyper_model.epochs == 10
    assert hyper_model.device == 'cpu'

def test_HyperExpertNN_forward():
    torch.manual_seed(42)

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=10,
                                device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    answ = hyper_model.forward(X)

    assert answ.long().sum() == 1

def test_HyperExpertNN_E_step():
    torch.manual_seed(42)

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=10,
                                device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    hyper_model.E_step(X, Y, Z, HyperParameters)

    assert hyper_model.input_dim == 2
    assert hyper_model.output_dim == 2
    assert hyper_model.epochs == 10
    assert hyper_model.device == 'cpu'


def test_HyperExpertNN_E_step():
    torch.manual_seed(42)

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=100,
                                device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    hyper_model.M_step(X, Y, Z, HyperParameters)

    params = hyper_model.parameters()
    sums = 0.
    for param in params:
        sums += param.sum()

    assert sums.long() == 5


def test_HyperExpertNN_LogPiExpectation():
    torch.manual_seed(42)

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=100,
                                device = 'cpu')
    
    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    log_pi = hyper_model.LogPiExpectation(X, Y, HyperParameters)

    assert log_pi.sum().long().item() == -2

def test_HyperExpertNN_PredictPi():
    torch.manual_seed(42)

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=100,
                                device = 'cpu')
    
    X = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    pi = hyper_model.PredictPi(X, HyperParameters)

    assert pi.sum().long().item() == 2




def test_HyperModelGateSparsed_init():
    torch.manual_seed(42)

    hyper_model = HyperModelGateSparsed(output_dim = 2, device = 'cpu')

    assert hyper_model.output_dim == 2
    assert hyper_model.device == 'cpu'
    assert (hyper_model.mu == 0.5*torch.ones(2)).all()
    assert hyper_model.gamma == 1.


def test_HyperModelGateSparsed_E_step():
    torch.manual_seed(42)

    hyper_model = HyperModelGateSparsed(output_dim = 2, device = 'cpu')

    
    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    hyper_model.E_step(X, Y, Z, HyperParameters)

    assert hyper_model.output_dim == 2
    assert hyper_model.device == 'cpu'
    assert (hyper_model.mu == 0.5*torch.ones(2)).all()
    assert (torch.round(hyper_model.mu_posterior) == 
            torch.tensor([[1., 0.], [1., 0.]])).all()
    assert (torch.round(hyper_model.gamma_posterior) == 
            torch.tensor([[3.], [2.]])).all()

def test_HyperModelGateSparsed_M_step():
    torch.manual_seed(42)

    hyper_model = HyperModelGateSparsed(output_dim = 2, device = 'cpu')

    
    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    hyper_model.M_step(X, Y, Z, HyperParameters)

    assert hyper_model.output_dim == 2
    assert hyper_model.device == 'cpu'
    assert (hyper_model.mu == 0.5*torch.ones(2)).all()
    assert hyper_model.gamma == 1.

def test_HyperModelGateSparsed_LogPiExpectation():
    torch.manual_seed(42)

    hyper_model = HyperModelGateSparsed(output_dim = 2, device = 'cpu')

    
    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    HyperParameters = {'beta': torch.tensor(1.)}

    log_pi = hyper_model.LogPiExpectation(X, Y, HyperParameters)

    assert log_pi.sum().long().item() == -5.

def test_HyperModelGateSparsed_PredictPi():
    torch.manual_seed(42)

    hyper_model = HyperModelGateSparsed(output_dim = 2, device = 'cpu')

    X = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.)}

    pi = hyper_model.PredictPi(X, HyperParameters)

    assert pi.sum().long().item() == 2

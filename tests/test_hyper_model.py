#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

from mixturelib.hyper_models import HyperModel
from mixturelib.hyper_models import HyperModelDirichlet
from mixturelib.hyper_models import HyperExpertNN

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


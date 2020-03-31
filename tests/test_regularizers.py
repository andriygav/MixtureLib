#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import torch

from mixturelib.regularizers import Regularizers
from mixturelib.regularizers import RegularizeModel
from mixturelib.regularizers import RegularizeFunc
from mixturelib.local_models import EachModelLinear

def test_Regularizers():
    model = Regularizers()

    with pytest.raises(NotImplementedError):
        model.E_step(None, None, None, None)

    with pytest.raises(NotImplementedError):
        model.M_step(None, None, None, None)

def test_RegularizeModel_init():
    model = RegularizeModel()
    assert model.ListOfModels == []

    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeModel(ListOfModels = list_of_models, device = 'cpu')

    assert len(model.ListOfModelsW0) == 0

    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = torch.tensor([0., 0.]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeModel(ListOfModels = list_of_models, device = 'cpu')

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()

def test_RegularizeModel_E_step():
    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeModel(ListOfModels = list_of_models, device = 'cpu')

    assert len(model.ListOfModelsW0) == 0

    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = torch.tensor([0., 0.]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeModel(ListOfModels = list_of_models, device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.), 
                       'alpha': torch.tensor(1.)}

    model.E_step(X, Y, Z, HyperParameters)

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()


def test_RegularizeModel_M_step_diag_w_diag_A():
    torch.manual_seed(42)
    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = torch.tensor([[0.], [0.]]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeModel(ListOfModels = list_of_models, device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.), 
                       'alpha': torch.tensor([1., 1.])}

    model.M_step(X, Y, Z, HyperParameters)

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()

def test_RegularizeModel_M_step_diag_w_mat_A():
    torch.manual_seed(42)
    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.zeros(2),
                            w = torch.tensor([[0.], [0.]]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeModel(ListOfModels = list_of_models, device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.), 
                       'alpha': torch.tensor(1.)}

    model.M_step(X, Y, Z, HyperParameters)

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()


def test_RegularizeFunc_init():
    model = RegularizeFunc()
    assert model.ListOfModels == []
    
    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = None, w = None, 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeFunc(ListOfModels = list_of_models, 
                            R = lambda x: x.sum(), epoch=100, device = 'cpu')

    assert len(model.ListOfModelsW0) == 0

    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = torch.tensor([0., 0.]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeFunc(ListOfModels = list_of_models, 
                            R = lambda x: x.sum(), epoch=100, device = 'cpu')

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()

    assert model.R(torch.ones(10, 10)) == 100
    assert model.epoch == 100

def test_RegularizeFunc_M_step_diag_w_diag_A():
    torch.manual_seed(42)
    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = torch.tensor([0., 0.]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeFunc(ListOfModels = list_of_models, 
                            R = lambda x: x.sum(), epoch=100, device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.), 
                       'alpha': torch.tensor([1., 1.])}

    model.M_step(X, Y, Z, HyperParameters)

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()

def test_RegularizeFunc_M_step_diag_w_mat_A():
    torch.manual_seed(42)
    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.zeros(2),
                            w = torch.tensor([0., 0.]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeFunc(ListOfModels = list_of_models, 
                            R = lambda x: x.sum(), epoch=100, device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.), 
                       'alpha': torch.tensor(1.)}

    model.M_step(X, Y, Z, HyperParameters)

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()



def test_RegularizeFunc_E_step():
    torch.manual_seed(42)
    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = torch.tensor([0., 0.]), 
                            OptimizedHyper = set(['w_0', 'A', 'beta'])))

    model = RegularizeFunc(ListOfModels = list_of_models, 
                            R = lambda x: x.sum(), epoch=100, device = 'cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)
    Z = torch.randn(2, 2)
    HyperParameters = {'beta': torch.tensor(1.), 
                       'alpha': torch.tensor(1.)}

    model.E_step(X, Y, Z, HyperParameters)

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()




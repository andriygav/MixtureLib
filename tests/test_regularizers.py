#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

from mixturelib.regularizers import Regularizers
from mixturelib.regularizers import RegularizeModel
from mixturelib.regularizers import RegularizeFunc

from mixturelib.local_models import EachModelLinear

def test_RegularizeModel_init():
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


def test_RegularizeModel_M_step():
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
                       'alpha': torch.tensor(1.)}

    model.M_step(X, Y, Z, HyperParameters)

    assert len(model.ListOfModelsW0) == 2
    assert model.ListOfModelsW0[0][0] == 0
    assert (model.ListOfModelsW0[0][1] == list_of_models[0].w_0).all()
    assert model.ListOfModelsW0[1][0] == 1
    assert (model.ListOfModelsW0[1][1] == list_of_models[1].w_0).all()


def test_RegularizeFunc_init():
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


def test_RegularizeFunc_M_step():
    torch.manual_seed(42)
    list_of_models = []
    for _ in range(2):
        list_of_models.append(
            EachModelLinear(input_dim = 2, device = 'cpu', 
                            A = torch.tensor([1., 1.]), 
                            w = torch.tensor([[0.], [0.]]), 
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





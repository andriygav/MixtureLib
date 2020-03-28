#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

from mixturelib.mixture import Mixture
from mixturelib.mixture import MixtureEM
from mixturelib.local_models import EachModelLinear
from mixturelib.hyper_models import HyperModelDirichlet
from mixturelib.hyper_models import HyperExpertNN
from mixturelib.regularizers import RegularizeFunc


def test_MixtureEM_sample_init():
    torch.random.manual_seed(42)
    HyperParameters = {'beta': 1.}

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=10,
                                device = 'cpu')

    first_model = EachModelLinear(input_dim=2)
    secode_model = EachModelLinear(input_dim=2)

    list_of_models = [first_model, secode_model]

    list_regulizer = [RegularizeFunc(ListOfModels=list_of_models)]

    mixture = MixtureEM(HyperParameters=HyperParameters,
                        HyperModel=hyper_model, 
                        ListOfModels=list_of_models, 
                        ListOfRegularizeModel=list_regulizer,
                        model_type='sample', 
                        device='cpu')

    assert mixture.K == 2
    assert mixture.device == 'cpu'
    assert mixture.HyperParameters['beta'] == torch.tensor(1.)
    assert mixture.HyperModel == hyper_model
    assert mixture.ListOfRegularizeModel[0] == list_regulizer[0]
    assert len(mixture.ListOfModels) == len(list_of_models)
    assert mixture.pZ is None

def test_MixtureEM_sample_E_step():
    torch.random.manual_seed(42)
    HyperParameters = {'beta': 1.}

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=10,
                                device = 'cpu')

    first_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))
    secode_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))

    list_of_models = [first_model, secode_model]

    list_regulizer = [RegularizeFunc(ListOfModels=list_of_models)]

    mixture = MixtureEM(HyperParameters=HyperParameters,
                        HyperModel=hyper_model,
                        ListOfModels=list_of_models,
                        model_type='sample',  
                        device='cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)


    mixture.E_step(X, Y)

    assert mixture.pZ.long().sum().item() == 0
    assert mixture.temp2_proba.long().sum().item() == 0
    assert mixture.indexes.long().sum().item() == 0
    assert mixture.lerning_indexes[0][0].item() == 0
    assert mixture.lerning_indexes[0][1].item() == 1
    assert mixture.lerning_indexes[1][0].item() == 0
    assert mixture.lerning_indexes[0][1].item() == 1


def test_MixtureEM_sample_E_step():
    torch.random.manual_seed(42)
    HyperParameters = {'beta': 1.}

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=10,
                                device = 'cpu')

    first_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))
    secode_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))

    list_of_models = [first_model, secode_model]

    list_regulizer = [RegularizeFunc(ListOfModels=list_of_models)]

    mixture = MixtureEM(HyperParameters=HyperParameters,
                        HyperModel=hyper_model,
                        ListOfModels=list_of_models,
                        ListOfRegularizeModel=list_regulizer,
                        model_type='sample',
                        device='cpu')

    X = torch.randn(200, 2)
    Y = torch.randn(200, 1)


    mixture.E_step(X, Y)
    mixture.M_step(X, Y)


def test_MixtureEM_sample_fit_predict():
    torch.random.manual_seed(42)
    HyperParameters = {'beta': 1.}

    hyper_model = HyperExpertNN(input_dim = 2,
                                hidden_dim = 2,
                                output_dim = 2,
                                epochs=10,
                                device = 'cpu')

    first_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))
    secode_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))

    list_of_models = [first_model, secode_model]

    list_regulizer = [RegularizeFunc(ListOfModels=list_of_models)]

    mixture = MixtureEM(HyperParameters=HyperParameters,
                        HyperModel=hyper_model,
                        ListOfModels=list_of_models,
                        ListOfRegularizeModel=list_regulizer,
                        model_type='sample', 
                        device='cpu')

    X = torch.randn(20, 2)
    Y = torch.randn(20, 1)


    mixture.fit(X, Y)

    answ, pi = mixture.predict(X)

    assert answ.sum().long() == -3
    assert pi.sum() == 20


def test_MixtureEM_init():
    torch.random.manual_seed(42)
    HyperParameters = {'beta': 1.}

    hyper_model = HyperModelDirichlet(
        output_dim = 2, device = 'cpu')

    first_model = EachModelLinear(input_dim=2)
    secode_model = EachModelLinear(input_dim=2)

    list_of_models = [first_model, secode_model]

    list_regulizer = [RegularizeFunc(ListOfModels=list_of_models)]

    mixture = MixtureEM(HyperParameters=HyperParameters,
                        HyperModel=hyper_model, 
                        ListOfModels=list_of_models, 
                        ListOfRegularizeModel=list_regulizer, 
                        device='cpu')

    assert mixture.K == 2
    assert mixture.device == 'cpu'
    assert mixture.HyperParameters['beta'] == torch.tensor(1.)
    assert mixture.HyperModel == hyper_model
    assert mixture.ListOfRegularizeModel[0] == list_regulizer[0]
    assert len(mixture.ListOfModels) == len(list_of_models)
    assert mixture.pZ is None

def test_MixtureEM_E_step():
    torch.random.manual_seed(42)
    HyperParameters = {'beta': 1.}

    hyper_model = HyperModelDirichlet(
        output_dim = 2, device = 'cpu')

    first_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))
    secode_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))

    list_of_models = [first_model, secode_model]

    list_regulizer = [RegularizeFunc(ListOfModels=list_of_models)]

    mixture = MixtureEM(HyperParameters=HyperParameters,
                        HyperModel=hyper_model, 
                        ListOfModels=list_of_models,  
                        device='cpu')

    X = torch.randn(2, 2)
    Y = torch.randn(2, 1)


    mixture.E_step(X, Y)

    assert mixture.pZ.long().sum().item() == 0
    assert mixture.temp2_proba.long().sum().item() == 0
    assert mixture.indexes.long().sum().item() == 0
    assert mixture.lerning_indexes[0][0].item() == 0
    assert mixture.lerning_indexes[0][1].item() == 1
    assert mixture.lerning_indexes[1][0].item() == 0
    assert mixture.lerning_indexes[0][1].item() == 1


def test_MixtureEM_E_step():
    torch.random.manual_seed(42)
    HyperParameters = {'beta': 1.}

    hyper_model = HyperModelDirichlet(
        output_dim = 2, device = 'cpu')

    first_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))
    secode_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))

    list_of_models = [first_model, secode_model]

    list_regulizer = [RegularizeFunc(ListOfModels=list_of_models)]

    mixture = MixtureEM(HyperParameters=HyperParameters,
                        HyperModel=hyper_model, 
                        ListOfModels=list_of_models, 
                        ListOfRegularizeModel=list_regulizer, 
                        device='cpu')

    X = torch.randn(200, 2)
    Y = torch.randn(200, 1)


    mixture.E_step(X, Y)
    mixture.M_step(X, Y)


def test_MixtureEM_fit_predict():
    torch.random.manual_seed(42)
    HyperParameters = {'beta': 1.}

    hyper_model = HyperModelDirichlet(
        output_dim = 2, device = 'cpu')

    first_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))
    secode_model = EachModelLinear(
        input_dim=2, 
        w=torch.tensor([.0, 0.]), 
        A=torch.tensor([1., 1.]))

    list_of_models = [first_model, secode_model]

    list_regulizer = [RegularizeFunc(ListOfModels=list_of_models)]

    mixture = MixtureEM(HyperParameters=HyperParameters,
                        HyperModel=hyper_model, 
                        ListOfModels=list_of_models, 
                        ListOfRegularizeModel=list_regulizer, 
                        device='cpu')

    X = torch.randn(20, 2)
    Y = torch.randn(20, 1)


    mixture.fit(X, Y)

    answ, pi = mixture.predict(X)

    assert answ.sum().long() == 0
    assert pi.sum() == 20

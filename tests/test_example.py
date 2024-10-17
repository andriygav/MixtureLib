#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np

from mixturelib.mixture import MixtureEM
from mixturelib.local_models import EachModelLinear
from mixturelib.hyper_models import HyperExpertNN, HyperModelDirichlet

np.random.seed(42)

N = 200
noise_component = 0.8
noise_target = 5

X = np.random.randn(N, 2)
X[:N//2, 1] *= noise_component
X[N//2:, 0] *= noise_component

real_first_w = np.array([[10.], [0.]])
real_second_w = np.array([[0.], [30.]])

y = np.vstack([X[:N//2]@real_first_w, X[N//2:]@real_second_w])\
    + noise_target*np.random.randn(N, 1)

torch.random.manual_seed(42)
X_tr = torch.FloatTensor(X)
Y_tr = torch.FloatTensor(y)

def test_example_mixture_of_model():
    torch.random.manual_seed(42)
    first_model = EachModelLinear(input_dim=2)
    secode_model = EachModelLinear(input_dim=2)

    list_of_models = [first_model, secode_model]

    HpMd = HyperModelDirichlet(output_dim=2)

    mixture = MixtureEM(HyperParameters={'beta': 1.},
                        HyperModel=HpMd,
                        ListOfModels=list_of_models,
                        model_type='sample')

    mixture.fit(X_tr, Y_tr)

    predicted_first_w = mixture.ListOfModels[0].W.numpy()
    predicted_second_w = mixture.ListOfModels[1].W.numpy()

    assert (predicted_first_w > 0.).all()
    assert (predicted_second_w > 0.).all()

def test_example_mixture_of_experts():
    torch.random.manual_seed(42)
    first_model = EachModelLinear(input_dim=2)
    secode_model = EachModelLinear(input_dim=2)

    list_of_models = [first_model, secode_model]

    HpMd = HyperExpertNN(input_dim=2, hidden_dim=5,
                     output_dim=2, epochs=20)

    mixture = MixtureEM(HyperParameters={'beta': 1.},
                        HyperModel=HpMd,
                        ListOfModels=list_of_models,
                        model_type='sample')

    mixture.fit(X_tr, Y_tr)

    predicted_first_w = mixture.ListOfModels[0].W.numpy()
    predicted_second_w = mixture.ListOfModels[1].W.numpy()

    assert (predicted_first_w > 0.).all()
    assert (predicted_second_w > 0.).all()

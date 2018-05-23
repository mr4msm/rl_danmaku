# -*- coding: utf-8 -*-
from chainer import Chain
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainerrl.agents.a3c import A3CModel
from chainerrl.policies import SoftmaxPolicy


class BaseModel(Chain):
    def __init__(self, in_channels, n_actions):
        super(BaseModel, self).__init__()
        initializer = initializers.HeNormal()

        with self.init_scope():
            self.conv0 = L.Convolution2D(in_channels, 8, 4,
                                         stride=2, pad=1,
                                         initialW=initializer)
            self.conv1 = L.Convolution2D(8, 16, 4,
                                         stride=2, pad=1,
                                         initialW=initializer)
            self.conv2 = L.Convolution2D(16, 32, 4,
                                         stride=2, pad=1,
                                         initialW=initializer)
            self.conv3 = L.Convolution2D(32, 64, 4,
                                         stride=2, pad=1,
                                         initialW=initializer)
            self.fc4 = L.Linear(None, 256, initialW=initializer)
            self.fc5 = L.Linear(256, n_actions, initialW=initializer)

    def __call__(self, x):
        y = F.relu(self.conv0(x))
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = F.relu(self.fc4(y))
        y = self.fc5(y)

        return y


class Model(Chain, A3CModel):
    def __init__(self, in_channels, n_actions):
        super(Model, self).__init__()

        with self.init_scope():
            self.pi = SoftmaxPolicy(
                model=BaseModel(in_channels, n_actions))
            self.v = BaseModel(in_channels, 1)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)

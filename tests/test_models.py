from unittest import TestCase
import numpy as np

import nn
from nn.npops import np_conv, np_pad


class TestModels(TestCase):
    def test_simple_conv(self):
        bs = 10
        r = 2
        n_channels = 3
        n_filters = 5
        w, h = 32, 32

        # define a simple model
        x = nn.Tensor((bs, n_channels, h, w))
        W = nn.Constant(np.random.randn(n_filters, n_channels, 2 * r + 1, 2 * r + 1).astype(np.float32))
        y = nn.conv(x, W)

        # check shapes
        self.assertEqual(y.output.shape, (bs, n_filters, h, w))

        # compile model
        mdl = nn.compile_model(x, y)

        # emulate model with numpy
        npx = np.random.randn(bs, n_channels, h, w).astype(np.float32)
        npy = np_pad(npx, r)
        npy = np_conv(npy, W.value)

        # results must be very similar
        res = mdl(npx)
        self.assertLess(np.max(np.abs(npy - res)), 1e-4)

    def test_two_layer_conv(self):
        bs = 10
        r = 2
        n_channels = 3
        n_filters = 5
        w, h = 32, 32

        # define a simple model
        x = nn.Tensor((bs, n_channels, h, w))
        W1 = nn.Constant(np.random.randn(n_filters, n_channels, 2 * r + 1, 2 * r + 1).astype(np.float32))
        W2 = nn.Constant(np.random.randn(n_channels, n_filters, 2 * r + 1, 2 * r + 1).astype(np.float32))
        y = nn.conv(x, W1)
        self.assertEqual(y.output.shape, (bs, n_filters, h, w))

        y = nn.conv(y, W2)
        self.assertEqual(y.output.shape, (bs, n_channels, h, w))

        # compile model
        mdl = nn.compile_model(x, y)

        # emulate model with numpy
        npx = np.random.randn(bs, n_channels, h, w).astype(np.float32)
        npy = np_pad(npx, r)
        npy = np_conv(npy, W1.value)
        npy = np_pad(npy, r)
        npy = np_conv(npy, W2.value)

        # results must be very similar
        res = mdl(npx)
        self.assertLess(np.max(np.abs(npy - res)), 1e-3)

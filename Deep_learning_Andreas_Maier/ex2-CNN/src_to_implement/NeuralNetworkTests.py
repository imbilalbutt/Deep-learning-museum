import unittest
from Layers import *
from Optimization import *
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import tabulate
import argparse

ID = 2  # identifier for dispatcher


class TestFullyConnected2(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros(shape)
            weights[0] = 1
            weights[1] = 2
            return weights

    def test_trainable(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue(layer.trainable, "Possible reason: FullyConnected Layer doesn't inherit the Base Layer and/or"
                                         "attribute trainable isn't set in the FullyConnected Layer.")
    
    def test_weights_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue((layer.weights.shape) in ((self.input_size + 1, self.output_size), (self.output_size, self.input_size + 1)),
                        "Possible reason: Bias has not been appended to the weights.")

    def test_forward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size, "Possible reason: Weights have the wrong shape or"
                                                                   "the formula isn't implemented correctly.")
        self.assertEqual(output_tensor.shape[0], self.batch_size, "Possible reason: Shape of the input_tensor is"
                                                                  "getting changed in the wrong dimension or the"
                                                                  "formula is not implemented correctly.")

    def test_backward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size, "Possible reason: Bias column has not been removed,"
                                                                 "the weights have not been transposed, or the formula"
                                                                 "is not implemented correctly.")
        self.assertEqual(error_tensor.shape[0], self.batch_size, "Possible reason: error_tensor has been changed or the"
                                                                 "formula is not implemented correctly.")

    def test_update(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([ self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)),
                            "The learning process doesn't work properly, check the gradient w.r.t. the weights and make"
                            "sure, the weights get updated correctly.")

    def test_update_bias(self):
        input_tensor = np.zeros([self.batch_size, self.input_size])
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)),
                            "Possible reason: Handling of the bias is not correct. Make sure to append the bias to the"
                            "weight matrix in the correct place and include the bias when updating the weights.")

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5, "Possible reason: Calculation of the gradient w.r.t. the input"
                                                       "is incorrect.")

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5, "Possible reason: Calculation of the gradient w.r.t the weights"
                                                       "is incorrect.")

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected.FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0, "Possible reason: Bias is not appended to the weights correctly, or in"
                                              "the forward pass the bias column is not appended to the input_tensor.")

    def test_initialization(self):
        input_size = 4
        categories = 10
        layer = FullyConnected.FullyConnected(input_size, categories)
        init = TestFullyConnected2.TestInitializer()
        layer.initialize(init, Initializers.Constant(0.5))
        self.assertEqual(init.fan_in, input_size, "Possible reason: Shape of the weights is not correct, check their"
                                                  "initialization.")
        self.assertEqual(init.fan_out, categories, "Possible reason: Shape of the weights is not correct, check their"
                                                   "initialization.")
        if layer.weights.shape[0]>layer.weights.shape[1]:
            self.assertLessEqual(np.sum(layer.weights) - 17, 1e-5, "Possible reason: Weights and/or bias is not"
                                                                   "initialized correctly, check their values.")
        else:
            self.assertLessEqual(np.sum(layer.weights) - 35, 1e-5, "Possible reason: Weights and/or bias is not"
                                                                   "initialized correctly, check their values.")


class TestReLU(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size, :] -= 2

        self.label_tensor = np.zeros([self.batch_size, self.input_size])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_trainable(self):
        layer = ReLU.ReLU()
        self.assertFalse(layer.trainable, "Possible reason: ReLU doesn't inherit from the base layer.")

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = ReLU.ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0,
                         "Possible reason: Forward pass in ReLU is not correct. Check the formula again.")

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = ReLU.ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor*2)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0,
                         "Possible reason: Backward pass in ReLU is not correct. Check the formula again.")

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(ReLU.ReLU())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5, "Possible reason: ReLU is not implemented correctly, check the"
                                                       "tests for the forward and backward pass.")


class TestSoftMax(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = SoftMax.SoftMax()
        self.assertFalse(layer.trainable, "Possible reason: SoftMax doesn't inherit from the base layer.")

    def test_forward_shift(self):
        input_tensor = np.zeros([self.batch_size, self.categories]) + 10000.
        layer = SoftMax.SoftMax()
        pred = layer.forward(input_tensor)
        self.assertFalse(np.isnan(np.sum(pred)), "Possible reason: Input tensor is not shifted by subtracting the max"
                                                 "value.")

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertLess(loss, 1e-10, "Possible reason: Formula for the forward method is not implemented correctly.")

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()
        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        self.assertAlmostEqual(np.sum(error), 0, msg="Possible reason: Formula for the backward pass is not implemented"
                                                     "correctly.")

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12, msg="Possible reason: Formula for the forward pass is not implemented"
                                                    "correctly for very wrong predictions.")

    def test_regression_backward_high_loss_w_CrossEntropy(self):
        input_tensor = self.label_tensor - 1
        input_tensor *= -10.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertAlmostEqual(element, 1/3, places=3, msg="Possible reason: Formula for SoftMax or"
                                                               "CrossEntropyLoss is not implemented correctly. Wrong"
                                                               "predictions are not punished.")

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1, places=3, msg="Possible reason: Formula for SoftMax or"
                                                              "CrossEntropyLoss is not implemented correctly. Correct"
                                                              "predictions are not strengthened.")

    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0., "Possible reason: You had very bad luck, try this test again. If it always"
                                            "fails, check if you changed the L2Loss class provided down below.")

    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertLessEqual(element, 0, "Possible reason: Formula for SoftMax is not implemented correctly. Wrong"
                                             "predictions are not punished.")

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertGreaterEqual(element, 0, "Possible reason: Formula for SoftMax is not implemented correctly."
                                                "Correct predictions are not strengthened.")

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(SoftMax.SoftMax())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5, "Possible reason: Forward or backward pass of SoftMax is not"
                                                       "implemented correctly, check the formulas again.")

    def test_predict(self):
        input_tensor = np.arange(self.categories * self.batch_size)
        input_tensor = input_tensor / 100.
        input_tensor = input_tensor.reshape((self.categories, self.batch_size))
        # print(input_tensor)
        layer = SoftMax.SoftMax()
        prediction = layer.forward(input_tensor.T)
        # print(prediction)
        expected_values = np.array([[0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724,
                                     0.21732724, 0.21732724],
                                    [0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387,
                                     0.23779387, 0.23779387],
                                    [0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794,
                                     0.26018794, 0.26018794],
                                    [0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095,
                                     0.28469095, 0.28469095]])
        # print(expected_values)
        # print(prediction)
        np.testing.assert_almost_equal(expected_values, prediction.T,
                                       err_msg="Possible reason: Forward pass of SoftMax is not implemented correctly,"
                                               "check the formula again.")


class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(Loss.CrossEntropyLoss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4, "Possible reason: If only this fails, backward pass of CELoss is"
                                                       "not implemented correctly, otherwise the forward pass might not"
                                                       "be implemented correctly as well. Check the formulas again.")

    def test_zero_loss(self):
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(self.label_tensor, self.label_tensor)
        self.assertAlmostEqual(loss, 0, msg="Possible reason: Forward pass of CELoss is not implemented correctly."
                                            "Check the formula again.")

    def test_high_loss(self):
        label_tensor = np.zeros((self.batch_size, self.categories))
        label_tensor[:, 2] = 1
        input_tensor = np.zeros_like(label_tensor)
        input_tensor[:, 1] = 1
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(input_tensor, label_tensor)
        self.assertAlmostEqual(loss, 324.3928805, places=4, msg="Possible reason: Forward pass of CELoss is not"
                                                                "implemented correctly. Check the formula again.")


class TestOptimizers2(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]),
                                       err_msg="Possible reason: Formula for the SGD is not implemented correctly.")

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]),
                                       err_msg="Possible reason: Formula for the SGD is not implemented correctly.")

    def test_sgd_with_momentum(self):
        optimizer = Optimizers.SgdWithMomentum(1., 0.9)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]),
                                       err_msg="Possible reason: Formula for the SGD with momentum is not implemented"
                                               "correctly. Momentum gradient v might be initialized with anything other"
                                               "than 0.")

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.9]),
                                       err_msg="Possible reason: Formula for the SGD with momentum is not implemented"
                                               "correctly. Momentum gradient v might not be stored correctly.")

    def test_adam(self):
        optimizer = Optimizers.Adam(1., 0.01, 0.02)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]),
                                       err_msg="Possible reason: Formula for ADAM is not implemented correctly. Make"
                                               "sure that r and v are initialized with zeros.")

        result = optimizer.calculate_update(result, .5)
        np.testing.assert_almost_equal(result, np.array([-0.9814473195614205]),
                                       err_msg="Possible reason: Formula for ADAM is not implemented correctly. If you"
                                               "are sure that the implementation is correct, try to set eps=1e-8")


class TestInitializers(unittest.TestCase):
    class DummyLayer:
        def __init__(self, input_size, output_size):
            self.weights = []
            self.shape = (output_size, input_size)

        def initialize(self, initializer):
            self.weights = initializer.initialize(self.shape, self.shape[1], self.shape[0])

    def setUp(self):
        self.batch_size = 9
        self.input_size = 400
        self.output_size = 400
        self.num_kernels = 20
        self.num_channels = 20
        self.kernelsize_x = 41
        self.kernelsize_y = 41

    def _performInitialization(self, initializer):
        np.random.seed(1337)
        layer = TestInitializers.DummyLayer(self.input_size, self.output_size)
        layer.initialize(initializer)
        weights_after_init = layer.weights.copy()
        return layer.shape, weights_after_init

    def test_uniform_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.UniformRandom())

        self.assertEqual(weights_shape, weights_after_init.shape, "Possible reason: Weights in uniform random are"
                                                                  "initialized in the wrong shape. Make sure to use the"
                                                                  "weights_shape attribute for them.")

    def test_uniform_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.UniformRandom())

        p_value = stats.kstest(weights_after_init.flat, 'uniform', args=(0, 1)).pvalue
        self.assertGreater(p_value, 0.01, "Possible reason: Weights are not initialized correctly, make sure, that"
                                          "they have a uniform distribution between 0 and 1.")

    def test_xavier_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.Xavier())

        self.assertEqual(weights_shape, weights_after_init.shape, "Possible reason: Weights in Xavier are initialized"
                                                                  "in the wrong shape. Make sure to use the"
                                                                  "weights_shape attribute for them.")

    def test_xavier_distribution(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.Xavier())

        scale = np.sqrt(2) / np.sqrt(self.input_size + self.output_size)
        p_value = stats.kstest(weights_after_init.flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01, "Possible reason: Formula for Xavier initialization is not implemented"
                                          "correctly.")

    def test_he_shape(self):
        weights_shape, weights_after_init = self._performInitialization(Initializers.He())

        self.assertEqual(weights_shape, weights_after_init.shape, "Possible reason: Weights in He are initialized in"
                                                                  "the wrong shape. Make sure to use the"
                                                                  "weights_shape attribute for them.")

    def test_he_distribution(self):
        weights_before_init, weights_after_init = self._performInitialization(Initializers.He())

        scale = np.sqrt(2) / np.sqrt(self.input_size)
        p_value = stats.kstest(weights_after_init.flat, 'norm', args=(0, scale)).pvalue
        self.assertGreater(p_value, 0.01, "Possible reason: Formula for He initialization is not implemented correctly")


class TestFlatten(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.input_shape = (3, 4, 11)
        self.input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        self.input_tensor = self.input_tensor.reshape(self.batch_size, *self.input_shape)

    def test_trainable(self):
        layer = Flatten.Flatten()
        self.assertFalse(layer.trainable, "Possible reason: Flatten doesn't inherit from the base layer.")

    def test_flatten_forward(self):
        flatten = Flatten.Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        input_vector = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_vector = input_vector.reshape(self.batch_size, np.prod(self.input_shape))
        self.assertLessEqual(np.sum(np.abs(output_tensor-input_vector)), 1e-9,
                             "Possible reason: Input tensor is not reshaped correctly. Make sure to keep the batch"
                             "dimension and don't swap axes for the other dimensions, when you flatten them.")

    def test_flatten_backward(self):
        flatten = Flatten.Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        backward_tensor = flatten.backward(output_tensor)
        self.assertLessEqual(np.sum(np.abs(self.input_tensor - backward_tensor)), 1e-9,
                             "Possible reason: Error tensor is not reshaped correctly, make sure to store the correct"
                             "shape in the forward pass.")


class TestConv(unittest.TestCase):
    plot = False
    directory = 'plots/'

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.input_size = 14 * 10 * 3
        self.uneven_input_shape = (3, 11, 15)
        self.uneven_input_size = 15 * 11 * 3
        self.spatial_input_shape = np.prod(self.input_shape[1:])
        self.kernel_shape = (3, 5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 105
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        self.assertTrue(layer.trainable, "Possible reason: Convolutional Layer doesn't inherit the Base Layer and/or"
                                         "attribute trainable isn't set in the Convolutional Layer.")

    def test_forward_size(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]),
                         "Possible reason: Shape of the output_tensor is not correct. Make sure to have a same"
                         "correlation in the image dimensions and a valid correlation in the channel dimension. The"
                         "desired output shape is Batches x #kernels x spatialX x (spatialY).")

    def test_forward_size_stride(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, 4, 7),
                         "Possible reason: If test_forward_size fails as well, fix that one first. Then, if this still"
                         "fails, your subsampling doesn't work properly. It might help, to do the subsampling on"
                         "paper and compare it to the intermediate results of your code.")

    def test_forward_size_stride_uneven_image(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels + 1)
        input_tensor = np.array(range(int(np.prod(self.uneven_input_shape) * (self.batch_size + 1))), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size + 1, *self.uneven_input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, ( self.batch_size+1, self.num_kernels+1, 4, 8),
                         "Possible reason: If test_forward_size and/or test_forward_size_stride fail as well, fix those"
                         "first. If they don't, then your subsampling doesn't work for an uneven sized image. Think"
                         "about what happens to the last few pixels, when doing this subsampling.")

    def test_forward(self):
        np.random.seed(1337)
        conv = Conv.Conv((1, 1), (1, 3, 3), 1)
        conv.weights = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        input_tensor = np.random.random((1, 1, 10, 14))
        expected_output = gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertAlmostEqual(difference, 0., places=1,
                               msg="Possible reason: Implementation for the forward pass is not correct. Check the"
                                   "correlation again and make sure, that you correlate the input_tensor with the"
                                   "correct kernel.")

    def test_forward_multi_channel(self):
        np.random.seed(1337)
        maps_in = 2
        bias = 1
        conv = Conv.Conv((1, 1), (maps_in, 3, 3), 1)
        filter = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.weights = np.repeat(filter[None, ...], maps_in, axis=1)
        conv.bias = np.array([bias])
        input_tensor = np.random.random((1, maps_in, 10, 14))
        expected_output = bias
        for map_i in range(maps_in):
            expected_output = expected_output + gaussian_filter(input_tensor[0, map_i, :, :], 0.85, mode='constant',
                                                                cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor) / maps_in)
        self.assertAlmostEqual(difference, 0., places=1,
                               msg="Possible reason: If test_forward fails as well, fix that one first. Otherwise,"
                                   "your implementation for the valid correlation in channel dimension might not work"
                                   "correctly. Make sure to select the correct channel if you do a same correlation."
                                   "Also make sure to add the bias.")

    def test_forward_fully_connected_channels(self):
        np.random.seed(1337)
        conv = Conv.Conv((1, 1), (3, 3, 3), 1)
        conv.weights = (1. / 15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        tensor = np.random.random((1, 1, 10, 14))
        input_tensor = np.zeros((1, 3, 10, 14))
        input_tensor[:, 0] = tensor.copy()
        input_tensor[:, 1] = tensor.copy()
        input_tensor[:, 2] = tensor.copy()
        expected_output = 3 * gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertLess(difference, 0.2, "Possible reason: If test_forward and/or test_forward_multi_channel fail as"
                                         "well, fix those first. Otherwise, your implementation for the valid"
                                         "correlation in channel dimension might not work correctly. Make sure to"
                                         "select the correct channel if you do a same correlation. It might help, to"
                                         "do the correlation on paper to get a better understanding of it.")

    def test_1D_forward_size(self):
        conv = Conv.Conv([2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(3 * 15 * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape((self.batch_size, 3, 15))
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape,  (self.batch_size, self.num_kernels, 8),
                         "Possible reason: If any other tests for the forward pass fail, fix those first. Otherwise, 1D"
                         "correlation is not implemented correctly. Make sure to differentiate between the 1D and 2D"
                         "case in your forward pass.")

    def test_backward_size(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape),
                         "Possible reason: Shape of the error_tensor is not correct. Make sure to have a same"
                         "correlation in the image dimensions and a valid correlation in the channel dimension. The"
                         "desired output shape is batches x channels x spatialX x (spatialY). Also make sure to"
                         "reslice the kernels correctly and flip them, if necessary. It might help, to do the reslicing"
                         "on paper and compare your results to the ones from your code.")

    def test_backward_size_stride(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape),
                         "Possible reason: If test_backward_size fails as well, fix that one first. Otherwise, the"
                         "upsampling of the error_tensor might not work properly. It might help, to do the upsampling"
                         "on paper and compare your results.")

    def test_1D_backward_size(self):
        conv = Conv.Conv([2], (3, 3), self.num_kernels)
        input_tensor = np.array(range(45 * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape((self.batch_size, 3, 15))
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, 3, 15),
                         "Possible reason: If any other tests for the backward_size and/or the 1D forward pass fail,"
                         "fix those first. Otherwise, 1D convolution is not implemented correctly. Make sure to"
                         "differentiate between the 1D and 2D case in your backward pass.")

    def test_1x1_convolution(self):
        conv = Conv.Conv((1, 1), (3, 1, 1), self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]))
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape),
                         "Possible reason: 1x1 convolution doesn't work for the backward pass. If any tests for the"
                         "backward pass fail, fix those first. Otherwise, make sure, that the kernel size of 1x1 does"
                         "not cause any unexpected behaviour in your code.")

    def test_layout_preservation(self):
        conv = Conv.Conv((1, 1), (3, 3, 3), 1)
        conv.initialize(self.TestInitializer(), Initializers.Constant(0.0))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(np.squeeze(output_tensor) - input_tensor[:, 1, :, :])), 0.,
                               msg="Possible reason: If any other tests for the forward pass fail, fix those first."
                                   "Otherwise, in the forward pass, make sure that you store the results of the"
                                   "correlation in the correct places for the output_tensor.")

    def test_gradient(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 5e-2,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass fail, fix"
                             "those first. Otherwise, your gradient w.r.t. lower layers is not correct. Check the"
                             "SoftConvTests for more detailed information as well. Make sure, that you have implemented"
                             "the valid convolution across the channel dimension correctly. Also check, whether your"
                             "kernels are resliced and flipped correctly. If you pad the error_tensor, make sure that"
                             "the padding is correct.")

    def test_gradient_weights(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass fail, fix"
                             "those first. Otherwise your gradient w.r.t weights is not correct. Check the"
                             "SoftConvTests for more detailed information as well. Make sure, that you have implemented"
                             "the valid convolution across the channel dimension correctly. If you pad the input_tensor"
                             "make sure, that the padding is correct.")

    def test_gradient_weights_strided(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 36])
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((2, 2), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass and/or"
                             "test_gradient_weights fail, fix those first. Otherwise your gradient w.r.t weights is not"
                             "correct in the case of stride. Check the SoftConvTests for more detailed information as"
                             "well. Check, if the upsampling of the error_tensor works correctly.")

    def test_gradient_bias(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-5,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass fail, fix"
                             "those first. Otherwise, your gradient w.r.t bias is not correct. Check the SoftConvTests"
                             "for more detailed information as well. Make sure, that you sum up the error_tensor over"
                             "the correct dimensions, as described in the formula on slide 18 of the exercise slides.")

    def test_weights_init(self):
        # simply checks whether you have not initialized everything with zeros
        conv = Conv.Conv((1, 1), (100, 10, 10), 150)
        self.assertGreater(np.mean(np.abs(conv.weights)), 1e-3,
                           "Possible reason: Don't initialize your weights with zeros, use a random initialization"
                           "instead.")

    def test_bias_init(self):
        conv = Conv.Conv((1, 1), (1, 1, 1), 150 * 100 * 10 * 10)
        self.assertGreater(np.mean(np.abs(conv.bias)), 1e-3,
                           "Possible reason: Don't initialize your bias with zeros, use a random initialization"
                           "instead.")

    def test_gradient_stride(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 35])
        input_tensor = np.abs(np.random.random((2, 6, 5, 14)))
        layers = list()
        layers.append(Conv.Conv((1, 2), (6, 3, 3), 1))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass and/or the"
                             "test_gradient fail, fix those first. Otherwise, your gradient w.r.t. lower layers is not"
                             "correct in case of stride. Check the SoftConvTests for more detailed information as well."
                             "Make sure, that the upsampling of the error_tensor works correctly. Also check, if the"
                             "valid convolution across the channel dimension is working correctly for the forward and"
                             "backward pass.")

    def test_update(self):
        input_tensor = np.random.uniform(-1, 1, (self.batch_size, *self.input_shape))
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        conv.optimizer = Optimizers.Sgd(1)
        conv.initialize(Initializers.He(), Initializers.Constant(0.1))
        # conv.weights = np.random.rand(4, 3, 5, 8)
        # conv.bias = 0.1 * np.ones(4)
        for _ in range(10):
            output_tensor = conv.forward(input_tensor)
            error_tensor = np.zeros_like(output_tensor)
            error_tensor -= output_tensor
            conv.backward(error_tensor)
            new_output_tensor = conv.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)),
                            "Possible reason: If any other tests fail, fix those first. Make sure, that the weights and"
                            "the bias get updated with the respective optimizer. Also check the SoftConvTests to see,"
                            "whether your gradients are correct.")

    def test_initialization(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        init = TestConv.TestInitializer()
        conv.initialize(init, Initializers.Constant(0.1))
        self.assertEqual(init.fan_in, np.prod(self.kernel_shape),
                         "Possible reason: Calculation of fan_in in your initialize method is not correct.")
        self.assertEqual(init.fan_out, np.prod(self.kernel_shape[1:]) * self.num_kernels,
                         "Possible reason: Calculation of fan_out in your initialize method is not correct.")


class TestPooling(unittest.TestCase):
    plot = False
    directory = 'plots/'

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (2, 4, 7)
        self.input_size = np.prod(self.input_shape)

        np.random.seed(1337)
        self.input_tensor = np.random.uniform(-1, 1, (self.batch_size, *self.input_shape))

        self.categories = 12
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

        self.layers = list()
        self.layers.append(None)
        self.layers.append(Flatten.Flatten())
        self.layers.append(L2Loss())
        self.plot_shape = (self.input_shape[0], np.prod(self.input_shape[1:]))

    def test_trainable(self):
        layer = Pooling.Pooling((2, 2), (2, 2))
        self.assertFalse(layer.trainable, "Possible reason: Pooling doesn't inherit from the base layer.")

    def test_shape(self):
        layer = Pooling.Pooling((2, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 3])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0,
                         "Possible reason: Output tensor from forward pass in Pooling has the wrong shape. Make sure to"
                         "calculate the correct shape in case of even and odd dimensions.")

    def test_overlapping_shape(self):
        layer = Pooling.Pooling((2, 1), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 2, 6])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0,
                         "Possible reason: Output tensor from forward pass in Pooling has the wrong shape. Make sure to"
                         "include the stride in both dimensions into your computations.")

    def test_subsampling_shape(self):
        layer = Pooling.Pooling((3, 2), (2, 2))
        result = layer.forward(self.input_tensor)
        expected_shape = np.array([self.batch_size, 2, 1, 3])
        self.assertEqual(np.sum(np.abs(np.array(result.shape) - expected_shape)), 0,
                         "Possible reason: Output tensor from forward pass in Pooling has the wrong shape. Make sure to"
                         "include the stride in both dimensions into your computations.")

    def test_gradient_stride(self):
        self.layers[0] = Pooling.Pooling((2, 2), (2, 2))
        difference = Helpers.gradient_check(self.layers, self.input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6,
                             "Possible reason: If the tests for the forward pass fail as well, fix those first. If they"
                             "pass, your backward pass is not correct. Make sure, you write the error back to the"
                             "correct index, that you stored in the forward pass. It might help, to do this upsampling"
                             "on paper and compare your intermediate results to the ones from the backward pass.")

    def test_gradient_overlapping_stride(self):
        label_tensor = np.random.random((self.batch_size, 24))
        self.layers[0] = Pooling.Pooling((2, 1), (2, 2))
        difference = Helpers.gradient_check(self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6,
                             "Possible reason: If the tests for the forward pass fail as well, fix those first. If they"
                             "pass, your backward pass is not correct. Make sure, you write the error back to the"
                             "correct index, that you stored in the forward pass. It might help, to do this upsampling"
                             "on paper and compare your intermediate results to the ones from the backward pass.")

    def test_gradient_subsampling_stride(self):
        label_tensor = np.random.random((self.batch_size, 6))
        self.layers[0] = Pooling.Pooling((3, 2), (2, 2))
        difference = Helpers.gradient_check(self.layers, self.input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-6,
                             "Possible reason: If the tests for the forward pass fail as well, fix those first. If they"
                             "pass, your backward pass is not correct. Make sure, you write the error back to the"
                             "correct index, that you stored in the forward pass. It might help, to do this upsampling"
                             "on paper and compare your intermediate results to the ones from the backward pass.")

    def test_layout_preservation(self):
        pool = Pooling.Pooling((1, 1), (1, 1))
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = pool.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(output_tensor-input_tensor)), 0.,
                               "Possible reason: For a pooling region of 1x1 and a stride of 1 the values in your the"
                               "input_tensor get changed. Check, if you store the selected maximum at the correct place"
                               "of the output_tensor.")

    def test_expected_output_valid_edgecase(self):
        input_shape = (1, 3, 3)
        pool = Pooling.Pooling((2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array([[[[4]]], [[[13]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0,
                         "Possible reason: In the forward pass the pooling region doesn't start at [0,0] or is not"
                         "shifted according to the stride. It might help to do the pooling for this input on paper and"
                         "compare your intermediate values to the ones from the forward pass.")

    def test_expected_output(self):
        input_shape = (1, 4, 4)
        pool = Pooling.Pooling((2, 2), (2, 2))
        batch_size = 2
        input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
        input_tensor = input_tensor.reshape(batch_size, *input_shape)
        result = pool.forward(input_tensor)
        expected_result = np.array([[[[5.,  7.], [13., 15.]]], [[[21., 23.], [29., 31.]]]])
        self.assertEqual(np.sum(np.abs(result - expected_result)), 0,
                         "Possible reason: In the forward pass the wrong values are selected for the maximum or the"
                         "pooling region lies over the wrong areas. It might help to do the pooling for this input"
                         "on paper and compare your intermediate values to the ones from the forward pass.")


class TestNeuralNetwork2(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'

    def test_append_layer(self):
        # this test checks if your network actually appends layers, whether it copies the optimizer to these layers, and
        # whether it handles the initialization of the layer's weights
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1),
                                          Initializers.Constant(0.123),
                                          Initializers.Constant(0.123))
        fcl_1 = FullyConnected.FullyConnected(1, 1)
        net.append_layer(fcl_1)
        fcl_2 = FullyConnected.FullyConnected(1, 1)
        net.append_layer(fcl_2)

        self.assertEqual(len(net.layers), 2, "Possible reason: append_layer method in NeuralNetwork doesn't actually"
                                             "append the layers.")
        self.assertFalse(net.layers[0].optimizer is net.layers[1].optimizer,
                         "Possible reason: The optimizer is not deepcopied, when a trainable layer is appended.")
        self.assertTrue(np.all(net.layers[0].weights == 0.123),
                        "Possible reason: The initialize method is not called for trainable layers that get appended.")

    def test_data_access(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-4),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        out = net.forward()
        out2 = net.forward()

        self.assertNotEqual(out, out2, "Possible reason: Setup of the iris dataset was unsuccessful, check, if your"
                                       "sklearn package is installed correctly. Or the UniformRandom or Constant"
                                       "initializer do not work properly.")

    def test_iris_data(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-3),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(4000)
        if TestNeuralNetwork2.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork2.pdf"), transparent=True, bbox_inches='tight',
                        pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9, "Possible reason: If unittests for the layers fail, fix those first. The"
                                          "training with Fully Connected layers doesn't work properly. Check the"
                                          "gradients for those layers. Also make sure, that your weights and bias get"
                                          "updated correctly. If the other iris_data tests pass, something might be"
                                          "wrong with your SGD.")

    def test_iris_data_with_momentum(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.SgdWithMomentum(1e-3, 0.8),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork2.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Momentum')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork2_Momentum.pdf"), transparent=True,
                        bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9, "Possible reason: If unittests for the layers fail, fix those first. The"
                                          "training with Fully Connected layers doesn't work properly. Check the"
                                          "gradients for those layers. Also make sure, that your weights and bias get"
                                          "updated correctly. If the other iris_data tests pass, something might be"
                                          "wrong with your SGD with momentum.")

    def test_iris_data_with_adam(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork2.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using ADAM')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork2_ADAM.pdf"), transparent=True,
                        bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9, "Possible reason: If unittests for the layers fail, fix those first. The"
                                          "training with Fully Connected layers doesn't work properly. Check the"
                                          "gradients for those layers. Also make sure, that your weights and bias get"
                                          "updated correctly. If the other iris_data tests pass, something might be"
                                          "wrong with your ADAM.")

    def test_digit_data(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(5e-3, 0.98, 0.999),
                                          Initializers.He(),
                                          Initializers.Constant(0.1))
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 200
        num_kernels = 4

        net.data_layer = Helpers.DigitData(batch_size)
        net.loss_layer = Loss.CrossEntropyLoss()

        cl_1 = Conv.Conv(conv_stride_shape, convolution_shape, num_kernels)
        net.append_layer(cl_1)
        cl_1_output_shape = (*input_image_shape[1:], num_kernels)
        net.append_layer(ReLU.ReLU())

        pool = Pooling.Pooling((2, 2), (2, 2))
        pool_output_shape = (4, 4, 4)
        net.append_layer(pool)
        fcl_1_input_size = np.prod(pool_output_shape)

        net.append_layer(Flatten.Flatten())

        fcl_1 = FullyConnected.FullyConnected(fcl_1_input_size, int(fcl_1_input_size/2.))
        net.append_layer(fcl_1)

        net.append_layer(ReLU.ReLU())

        fcl_2 = FullyConnected.FullyConnected(int(fcl_1_input_size/2.), categories)
        net.append_layer(fcl_2)

        net.append_layer(SoftMax.SoftMax())

        net.train(200)

        if TestNeuralNetwork2.plot:
            description = 'on_digit_data'
            fig = plt.figure('Loss function for training a Convnet on the Digit dataset')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestConvNet_" + description + ".pdf"), transparent=True,
                        bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%',
                  file=f)
        print('\nOn the UCI ML hand-written digits dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
        self.assertGreater(accuracy, 0.5, "Possible reason: If unittests for the layers fail, fix those first. The"
                                          "training with your CNN doesn't work properly. Check whether the"
                                          "SoftConvTests give you only zeros in the gradient tensor differences. Also"
                                          "make sure, that your weights and bias get updated correctly.")


class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


if __name__ == "__main__":

    import sys
    if sys.argv[-1] == "Bonus":
        # sys.argv.pop()
        loader = unittest.TestLoader()
        bonus_points = {}
        tests = [TestOptimizers2, TestInitializers, TestFlatten, TestConv, TestPooling, TestFullyConnected2,
                 TestNeuralNetwork2]
        percentages = [8, 5, 2, 45, 15, 2, 23]
        total_points = 0
        for t, p in zip(tests, percentages):
            if unittest.TextTestRunner().run(loader.loadTestsFromTestCase(t)).wasSuccessful():
                bonus_points.update({t.__name__: ["OK", p]})
                total_points += p
            else:
                bonus_points.update({t.__name__: ["FAIL", p]})

        import time
        time.sleep(1)
        print("=========================== Statistics ===============================")
        exam_percentage = 3
        table = []
        for i, (k, (outcome, p)) in enumerate(bonus_points.items()):
            table.append([i, k, outcome, "0 / {} (%)".format(p) if outcome == "FAIL" else "{} / {} (%)".format(p, p),
                          "{:.3f} / 10 (%)".format(p / 100 * exam_percentage)])
        table.append([])
        table.append(["Ex2", "Total Achieved", "", "{} / 100 (%)".format(total_points),
                      "{:.3f} / 10 (%)".format(total_points * exam_percentage / 100)])

        print(tabulate.tabulate(table, headers=['Pos', 'Test', "Result", 'Percent in Exercise', 'Percent in Exam'],
                                tablefmt="github"))
    else:
        unittest.main()

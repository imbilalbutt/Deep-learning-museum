import unittest
from Layers import *
from Optimization import *
import numpy as np
import NeuralNetwork
import matplotlib.pyplot as plt
import tabulate
import argparse

ID = 1  # identifier for dispatcher

class TestFullyConnected1(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1  # one-hot encoded labels

    def test_trainable(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue(layer.trainable, msg="Possible error: The  trainable flag is not set to True. Please make sure"
                                             " to set set trainable=True.")

    def test_weights_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue((layer.weights.shape) in ((self.input_size + 1, self.output_size), (self.output_size, self.input_size + 1)))

    def test_forward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1],
                         self.output_size,
                         msg="Possible error: The shape of the output tensor is not correct. The correct shape is"
                             " Batch x N_Neurons. The second dimension of self.weights determines the number of "
                             "neurons. Please refer to the exercise slides to use the correct computation of the"
                             "output tensor using the weights and the input. Additionally, make sure you combined the "
                             "weight matrix and the bias properly and extended the input such that the computation "
                             "includes weight multiplication and bias addition.")
        self.assertEqual(output_tensor.shape[0],
                         self.batch_size,
                         msg="Possible error: The batch size of the output tensor is not equal to the batch size of the"
                             " input tensor. Please refer to the exercise slides to use the correct computation of the"
                             "output tensor using the weights and the input."
                         )

    def test_backward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        # print(output_tensor.shape)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1],
                         self.input_size,
                         msg="Possible error: The shape of the output tensor (backward function) is not correct. "
                             "Please make sure that you remove the fake errors that were comptet because of the added"
                             " collumn in the input (to include bias addition in the weight matrix multiplication)."
                         )
        self.assertEqual(error_tensor.shape[0],
                         self.batch_size,
                         msg="Possible error: The batch size of the output tensor is not equal to the batch size of the"
                             " input tensor (in the backward pass). Please refer to the exercise slides to use the "
                             "correct computation of the output tensor using the weights and the input."
                         )

    def test_update(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            # print(error_tensor.shape)
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)),
                            msg="Possible error: the weight update has not been performed correctly. Have a look at the"
                                "computation of gradient_weights. If the gradient_weights test passes, have a look at "
                                "the optimizer. Please also make sure that the weights are updated in the backward pass"
                                "if an optimizer is set!")

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
                            msg="Possible error: the update of the bias has not been performed correctly. Have a look at the"
                                "computation of gradient_weights. If the gradient_weights test passes, have a look at "
                                "the optimizer. Please also make sure that the weights are updated in the backward pass"
                                "if an optimizer is set!"
                            )

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference),
                             1e-5,
                             msg="Possible error: The gradient with respect to the input is not correct. Please refer "
                                 "to the exercise slides to use the correct computation of the output tensor using "
                                 "the weights and the input.")

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference),
                             1e-5,
                             msg="Possible error: The gradient with respect to the weights is not correct. Please refer "
                                 "to the exercise slides to use the correct computation of the output tensor using "
                                 "the weights and the input. Please also make sure that you implemented the "
                                 "gradients_weights property and store the gradient weights in this variable."
                             )

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected.FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0,
                           msg="Possible error: The initialization of the bias (i.e. the weights if stored in "
                               "single matric) may be wrong. Make sure bias and weights are initialized randomly"
                               " between 0 and 1!")


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
        self.assertFalse(layer.trainable,
                         msg="Possible error: Trainable flag is set to true. Make sure it is set to False.")

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = ReLU.ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0,
                         msg="Possible error: the ReLU function is not properly implemented. Make sure that the function"
                             "sets all negative values to zero and passes all positive values to the next layer"
                             " as they are according to ReLU(x) = max(0, x). ")

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = ReLU.ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor * 2)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0,
                         msg="Possible error: The derivative of the ReLU function is not correctly implemented. Please"
                             " refer to the lecture slides for the correct derivative. Hint: you may need the input "
                             "tensor X of the forward pass also in the backward pass...")

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(ReLU.ReLU())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5,
                             msg="Possible error: The derivative of the ReLU function is not correctly implemented. Please"
                             " refer to the lecture slides for the correct derivative. Hint: you may need the input "
                             "tensor X of the forward pass also in the backward pass...")

class TestSoftMax(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = SoftMax.SoftMax()
        self.assertFalse(layer.trainable,
                         msg="Possible error: Trainable flag is set to true. Make sure it is set to False.")

    def test_forward_shift(self):
        input_tensor = np.zeros([self.batch_size, self.categories]) + 10000.
        layer = SoftMax.SoftMax()
        pred = layer.forward(input_tensor)
        self.assertFalse(np.isnan(np.sum(pred)),
                         msg="Possible error: The input tensor is not shifted to the negative domain. Please make sure "
                             "shift the input linearly to the negative domain in order to keep numerical stability.")

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertLess(loss, 1e-10,
                        msg="Possible error: The forward function is not implemented correctly. Please refer to the "
                            "lecture slides for help. Hint: The output of the SoftMax function corresponds to a "
                            "probability distribution to with the probabilities for each label (usually) in a"
                            " classification task. So check if the sum of the output is equal to 1!")

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()
        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        self.assertAlmostEqual(np.sum(error), 0,
                               msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                   " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                   "Y from the forward pass also in the backward pass... "
                                   "The test also fails if the CrossEntropyLoss() function is not yet or wrong implemented.")

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12,
                               msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                   " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                   "Y from the forward pass also in the backward pass... "
                               )

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
            self.assertAlmostEqual(element, 1/3, places = 3,
                                   msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                       " The class confidence for wrong predicted lables is not decreased in the backward function."
                                       " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                       "Y from the forward pass also in the backward pass."
                                       "The test also fails if the CrossEntropyLoss() function is not yet or wrong implemented."
                                   )

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1, places = 3,
                                   msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                       " The class confidence for correct predicted lables is not increased in the backward function."
                                       " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                       "Y from the forward pass also in the backward pass."
                                       "The test also fails if the CrossEntropyLoss() function is not yet or wrong implemented."
                                   )


    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.,
                           msg="Possible error: The forward function is not implemented correctly. Please refer to the "
                            "lecture slides for help. Hint: The output of the SoftMax function corresponds to a "
                            "probability distribution to with the probabilities for each label (usually) in a"
                            " classification task. So, check if the sum of the output is equal to 1!")


    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertLessEqual(element, 0,
                                 msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                     " The class confidence for wrong predicted lables is not decreased in the backward function."
                                     " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                     "Y from the forward pass also in the backward pass."
                                 )

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertGreaterEqual(element, 0,
                                    msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                        " The class confidence for wrong predicted lables is not decreased in the backward function."
                                        " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                        "Y from the forward pass also in the backward pass."
                                    )

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(SoftMax.SoftMax())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5,
                             msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                 " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                        "Y from the forward pass also in the backward pass. Also make sure to do the "
                                 "necessary summation in the backward pass over the right axis."
                             )

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
                                       err_msg="Possible error: The forward function is not properly implemented. "
                                               "Please refer to the lecture slided for the correct function. make "
                                               "sure to do the necessary summation in the forward pass over the right "
                                               "axis")


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
        self.assertLessEqual(np.sum(difference), 1e-4,
                             msg="Possible error: The backward function is not computing the correct gradient. "
                                 "This test fails for wrong implemented forward or backward function. Make sure to "
                                 "include the epsilon value in both function for numerical stability. Additional hint:"
                                 " You may need the output Y of the forward tensor also in the backward pass...")

    def test_zero_loss(self):
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(self.label_tensor, self.label_tensor)
        self.assertAlmostEqual(loss, 0,
                               msg="Possible error: The forward function is not correctly implemented. Please refer to "
                                   "the lecture slides for the correct function. Hint: the label has to be multiplied"
                                   " with the negative log of the prediction -ylog(y').")

    def test_high_loss(self):
        label_tensor = np.zeros((self.batch_size, self.categories))
        label_tensor[:, 2] = 1
        input_tensor = np.zeros_like(label_tensor)
        input_tensor[:, 1] = 1
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(input_tensor, label_tensor)
        self.assertAlmostEqual(loss, 324.3928805, places = 4,
                               msg="Possible error: The forward function is not correctly implemented. Please refer to "
                                   "the lecture slides for the correct function. Hint: the label has to be multiplied"
                                   " with the negative log of the prediction -ylog(y')."
                               )


class TestOptimizers1(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]),
                                       err_msg="Possible error: The Sgd optimizer is not properly implemented. "
                                               "SGD is used by some other unittests. If these fail it could be caused "
                                               "by a wrong implementation of the SGD optimizer.")

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]),
                                       err_msg="Possible error: The Sgd optimizer is not properly implemented. "
                                               "SGD is used by some other unittests. If these fail it could be caused "
                                               "by a wrong implementation of the SGD optimizer."
                                       )


class TestNeuralNetwork1(unittest.TestCase):

    def test_append_layer(self):
        # this test checks if your network actually appends layers and whether it copies the optimizer to these layers
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1))
        fcl_1 = FullyConnected.FullyConnected(1, 1)
        net.append_layer(fcl_1)
        fcl_2 = FullyConnected.FullyConnected(1, 1)
        net.append_layer(fcl_2)

        self.assertEqual(len(net.layers), 2,
                         msg="Possible error: The append_layer function is not yet implemented or wrong implemented."
                             "Make sure that the NeuralNetwork class is able to add a layer and stores it in a list "
                             "called layers.")
        self.assertFalse(net.layers[0].optimizer is net.layers[1].optimizer,
                         msg="Possible error: The optimizer is not copied for each layer. Make sure to perform a "
                             "deepcopy of the optimizer and assign it to every trainable layer.")

    def test_data_access(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1))
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

        self.assertNotEqual(out, out2,
                            msg="Possible error: The Neural Network hat no access to the provided data. Make sure to "
                                "create an attribute data_layer in the constructor. Additionally, make sure that the "
                                "forward function calls the next() function of this attribute to get the next batch as"
                                " inout_tensor for the forward.")

    def test_iris_data(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-3))
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

        net.train(4000)
        plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
        plt.plot(net.loss, '-x')
        plt.show()

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)
        index_maximum = np.argmax(results, axis=1)
        one_hot_vector = np.zeros_like(results)
        for i in range(one_hot_vector.shape[0]):
            one_hot_vector[i, index_maximum[i]] = 1

        correct = 0.
        wrong = 0.
        for column_results, column_labels in zip(one_hot_vector, labels):
            if column_results[column_labels > 0].all() > 0:
                correct += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)
        print('\nOn the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
        self.assertGreater(accuracy, 0.8,
                           msg="Your network is not learning. Make sure that the gradients are computed correctly"
                               " in all layers and make sure that the weights are updated in the backward functions. "
                               "Have a look at the displayed loss curve.")


class L2Loss:
    def __init__(self):
        self.input_tensor = None

    def predict(self, input_tensor):
        return input_tensor

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)


if __name__ == '__main__':

    import sys
    if sys.argv[-1] == "Bonus":
        loader = unittest.TestLoader()
        bonus_points = {}
        tests = [TestCrossEntropyLoss, TestFullyConnected1, TestReLU, TestOptimizers1, TestNeuralNetwork1, TestSoftMax]
        percentages = [10, 45, 5, 5, 25, 10]
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
        exam_percentage = 1.5
        table = []
        for i, (k, (outcome, p)) in enumerate(bonus_points.items()):
            table.append([i, k, outcome, "0 / {} (%)".format(p) if outcome == "FAIL" else "{} / {} (%)".format(p, p), "{:.3f} / 10 (%)".format(p/100 * exam_percentage)])
        table.append([])
        table.append(["Ex1", "Total Achieved", "", "{} / 100 (%)".format(total_points), "{:.3f} / 10 (%)".format(total_points * exam_percentage / 100)])
        print(tabulate.tabulate(table, headers=['Pos', 'Test', "Result", 'Percent in Exercise', 'Percent in Exam'], tablefmt="github"))
    else:
        unittest.main()

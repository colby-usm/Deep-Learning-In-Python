from neural.Initializers import HeNormal
from neural.LossFunctions import SimpleCrossEntropy
import numpy as np
from neural.Perceptron import Perceptron
from neural.Activations import Relu, Softmax
from neural.NeuralLayer import NeuralLayer

class TestPerceptron:
    rng = np.random.default_rng()

    @staticmethod
    def test_random_initialization():
        size = np.uint16(TestPerceptron.rng.integers(1, 256, dtype=np.uint8).item())
        p = Perceptron(size)
        x = TestPerceptron.rng.random(size, dtype=np.float32)

        out = p(x)
        expected = np.dot(x, p.weights) + p.bias

        if np.isclose(out, expected):
            print(f"[PASS] test_random_initialization: output = {out}")
        else:
            print(f"[FAIL] test_random_initialization: output = {out}, expected = {expected}")

    @staticmethod
    def test_inputted_weights():
        weights = TestPerceptron.rng.random(32, dtype=np.float32)
        bias = TestPerceptron.rng.random(1, dtype=np.float32).item()
        x = TestPerceptron.rng.random(32, dtype=np.float32)

        p = Perceptron(weights.shape[0], (weights, bias))
        out = p(x)
        expected = np.dot(x, p.weights) + p.bias

        if np.isclose(out, expected):
            print(f"[PASS] test_inputted_weights: output = {out}")
        else:
            print(f"[FAIL] test_inputted_weights: output = {out}, expected = {expected}")


class TestActivations():
    rng = np.random.default_rng()

    @staticmethod
    def test_Relu():

        x = TestActivations.rng.random(32, dtype=np.float32)
        p = Perceptron(np.uint16(32))
        relu = Relu()

        out = relu(p(x))

        expected = max(0, np.dot(p.weights, x) + p.bias)

        if np.isclose(np.array(out), np.array(expected)):
            print(f"[PASS] test_Relu: output = {out}")
        else:
            print(f"[FAIL] test_Relu: output = {out}, expected = {expected}")


class TestNeuralLayer():
    rng = np.random.default_rng()

    @staticmethod
    def test_single_NeuralLayer():

        relu = Relu()
        x = TestNeuralLayer.rng.random(256, dtype=np.float32)

        nl = NeuralLayer((10, 256), initializer=HeNormal())

        out = relu(nl(x))
        print(f"[Neural Layer results] {out}")


    @staticmethod
    def test_simple_fc_network():

        relu = Relu()
        softmax = Softmax()
        ce = SimpleCrossEntropy()
        layer1 = NeuralLayer((128,64), initializer=HeNormal())
        layer2 = NeuralLayer((10,128), initializer=HeNormal())


        x = TestNeuralLayer.rng.random(64, dtype=np.float32)
        y_true = TestNeuralLayer.rng.random(10, dtype=np.float32)


        x = relu(layer1(x))
        y_pred = softmax(layer2(x))
        loss = ce(y_pred, y_true)

        print(f"[Simple FC network output] {y_pred}")
        print(f"[Simple FC network sum] {np.sum(y_pred)}")
        print(f"[Simple FC network loss] {loss}")



TestPerceptron.test_random_initialization()
TestPerceptron.test_inputted_weights()

TestActivations.test_Relu()

TestNeuralLayer.test_single_NeuralLayer()
TestNeuralLayer.test_simple_fc_network()

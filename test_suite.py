from neural.Initializers import HeNormal
from neural.LossFunctions import SimpleCrossEntropy
import numpy as np
from neural.Tensor import Tensor
from neural.Perceptron import Perceptron
from neural.Activations import Relu, Softmax
from neural.LinearLayer import LinearLayer

class TestPerceptron:
    rng = np.random.default_rng()

    @staticmethod
    def test_random_initialization():
        size = np.uint16(TestPerceptron.rng.integers(1, 256, dtype=np.uint8).item())

        weights = Tensor(TestPerceptron.rng.random(size, dtype=np.float32))
        bias = Tensor(0.0, dtype=np.float32) 
        p = Perceptron((weights, bias))

        x = Tensor(TestPerceptron.rng.random(size, dtype=np.float32))

        out = p(x)
        expected = np.dot(x.data, weights.data) + bias.data

        if np.isclose(out.data, expected):
            print(f"[PASS] test_random_initialization: output = {out.data}")
        else:
            print(f"[FAIL] test_random_initialization: output = {out.data}, expected = {expected}")

    @staticmethod
    def test_inputted_weights():
        weights = Tensor(TestPerceptron.rng.random(32, dtype=np.float32))
        bias = Tensor(TestPerceptron.rng.random(1, dtype=np.float32).item())
        x = Tensor(TestPerceptron.rng.random(32, dtype=np.float32))

        p = Perceptron((weights, bias))
        out = p(x)
        expected = np.dot(x.data, weights.data) + bias.data

        if np.isclose(out.data, expected):
            print(f"[PASS] test_inputted_weights: output = {out.data}")
        else:
            print(f"[FAIL] test_inputted_weights: output = {out.data}, expected = {expected}")


class TestActivations:
    rng = np.random.default_rng()

    @staticmethod
    def test_Relu():
        w = Tensor(TestActivations.rng.random(32, dtype=np.float32))
        b = Tensor(TestActivations.rng.random(1, dtype=np.float32))
        x = Tensor(TestActivations.rng.random(32, dtype=np.float32))

        p = Perceptron((w,b))  # Perceptron expects a Tensor-compatible shape
        relu = Relu()

        out = relu(p(x))

        expected = np.maximum(0, np.dot(p.weights.data, x.data) + p.bias.data)

        if np.allclose(out.data, expected):
            print(f"[PASS] test_Relu: output = {out.data}")
        else:
            print(f"[FAIL] test_Relu: output = {out.data}, expected = {expected}")


class TestLinearLayer:
    rng = np.random.default_rng()

    @staticmethod
    def test_single_LinearLayer():
        relu = Relu()

        x = Tensor(TestLinearLayer.rng.random(256, dtype=np.float32))
 
        weights = Tensor(HeNormal()((10, 256)))
        biases = Tensor(HeNormal()((10,)))

        nl = LinearLayer((weights, biases))

        out = relu(nl(x))
        print(f"[Linear Layer results] {out.data}")

    @staticmethod
    def test_simple_fc_network():
        relu = Relu()
        softmax = Softmax()
        ce = SimpleCrossEntropy()

        # Initialize layers
        w1, b1 = Tensor(HeNormal()((128, 64))), Tensor(HeNormal()((128,)))
        w2, b2 = Tensor(HeNormal()((10, 128))), Tensor(HeNormal()((10,)))

        layer1 = LinearLayer(data=(w1, b1))
        layer2 = LinearLayer(data=(w2, b2))

        # Input / target
        x = Tensor(TestLinearLayer.rng.random(64, dtype=np.float32))
        y_true = Tensor(TestLinearLayer.rng.random(10, dtype=np.float32))

        # Forward pass
        x = relu(layer1(x))
        y_pred = softmax(layer2(x))
        loss = ce(y_pred, y_true)

        print(f"[Simple FC network output] {y_pred.data}")
        print(f"[Simple FC network sum] {np.sum(y_pred.data)}")
        print(f"[Simple FC network loss] {loss}")



TestPerceptron.test_random_initialization()
TestPerceptron.test_inputted_weights()

TestActivations.test_Relu()

TestLinearLayer.test_single_LinearLayer()
TestLinearLayer.test_simple_fc_network()

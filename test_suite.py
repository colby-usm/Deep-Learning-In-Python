from graphviz import Digraph
from neural.Initializers import HeNormal
import numpy as np
from neural.Tensor import Tensor
from neural.Perceptron import Perceptron
from neural.LinearLayer import LinearLayer


class TestPerceptron:
    rng = np.random.default_rng()

    @staticmethod
    def test_random_initialization():
        size = np.uint16(TestPerceptron.rng.integers(1, 256, dtype=np.uint8).item())
        weights = Tensor(TestPerceptron.rng.random(size, dtype=np.float32))
        bias = Tensor([0.0])
        p = Perceptron((weights, bias))
        x = Tensor(TestPerceptron.rng.random(size, dtype=np.float32))
        out = p(x)
        expected = np.dot(x.data, weights.data) + bias.data
        if np.isclose(out.data, expected):
            print(f"[PASS] test_random_initialization: output = {out.data}")
        else:
            print(
                f"[FAIL] test_random_initialization: output = {out.data}, expected = {expected}"
            )

    @staticmethod
    def test_inputted_weights():
        weights = Tensor(TestPerceptron.rng.random(32, dtype=np.float32))
        bias = Tensor([float(TestPerceptron.rng.random(dtype=np.float32))])
        x = Tensor(TestPerceptron.rng.random(32, dtype=np.float32))
        p = Perceptron((weights, bias))
        out = p(x)
        expected = np.dot(x.data, weights.data) + bias.data
        if np.isclose(out.data, expected):
            print(f"[PASS] test_inputted_weights: output = {out.data}")
        else:
            print(
                f"[FAIL] test_inputted_weights: output = {out.data}, expected = {expected}"
            )


class TestActivations:
    rng = np.random.default_rng()

    @staticmethod
    def test_relu():
        w = Tensor(TestActivations.rng.random(32, dtype=np.float32))
        b = Tensor([float(TestPerceptron.rng.random(dtype=np.float32))])
        x = Tensor(TestActivations.rng.random(32, dtype=np.float32))
        p = Perceptron((w, b))
        out = p(x).relu()
        expected = np.maximum(0, np.dot(p.weights.data, x.data) + p.bias.data)
        if np.allclose(out.data, expected):
            print(f"[PASS] test_relu: output = {out.data}")
        else:
            print(f"[FAIL] test_relu: output = {out.data}, expected = {expected}")


class TestLinearLayer:
    rng = np.random.default_rng()

    @staticmethod
    def test_single_linear_layer():
        x = Tensor(TestLinearLayer.rng.random(256, dtype=np.float32))
        weights = Tensor(HeNormal()((10, 256)))
        biases = Tensor(HeNormal()((10,)))
        nl = LinearLayer((weights, biases))
        out = nl(x).relu()
        print(f"[Linear Layer results] {out.data}")

    @staticmethod
    def test_fcn_with_mse(make_graph=True, graph_name="sigmoid_test"):

        w1 = Tensor(HeNormal()((128, 1024)), name="w1")
        b1 = Tensor(HeNormal()((1024,)), name="b1")
        layer1 = LinearLayer((w1, b1))

        w2 = Tensor(HeNormal()((1024, 1024)), name="w2")
        b2 = Tensor(HeNormal()((1024,)), name="b2")
        layer2 = LinearLayer((w2, b2))

        w3 = Tensor(HeNormal()((1024, 1)), name="w3")
        b3 = Tensor(HeNormal()((1,)), name="b3")
        layer3 = LinearLayer((w3, b3))

        feat = Tensor(TestLinearLayer.rng.random(128, dtype=np.float32), name="x")

        x = layer1(feat).relu()
        x = layer2(x).relu()
        y_pred = layer3(x).sigmoid()

        y_true = Tensor(TestLinearLayer.rng.random(1, dtype=np.float32))
        loss = y_pred.mse(y_true)

        if make_graph:
            dag = loss.backward()
            g = draw_graph(dag)
            g.render(graph_name, format="png", view=True)

    @staticmethod
    def test_fcn_with_cce(make_graph=False, graph_name="cce_test"):

        w1 = Tensor(HeNormal()((128, 1024)), name="w1")
        b1 = Tensor(HeNormal()((1024,)), name="b1")
        layer1 = LinearLayer((w1, b1))

        w2 = Tensor(HeNormal()((1024, 1024)), name="w2")
        b2 = Tensor(HeNormal()((1024,)), name="b2")
        layer2 = LinearLayer((w2, b2))

        w3 = Tensor(HeNormal()((1024, 10)), name="w3")
        b3 = Tensor(HeNormal()((10,)), name="b3")
        layer3 = LinearLayer((w3, b3))

        feat = Tensor(TestLinearLayer.rng.random(128, dtype=np.float32), name="x")

        x = layer1(feat).relu()
        x = layer2(x).relu()
        y_pred = layer3(x).softmax()

        y_true = Tensor(TestLinearLayer.rng.random(10, dtype=np.float32))
        loss = y_pred.categorical_cross_entropy(y_true)

        if make_graph:
            dag = loss.backward()
            g = draw_graph(dag)
            g.render(graph_name, format="png", view=True)


def draw_graph(dag: list) -> Digraph:
    print(f"Drawing Graph of len: {len(dag)}")
    dot = Digraph()
    for i, node in enumerate(dag):
        print(f"At Node {i}/{len(dag) - 1} {node}")
        nid = str(id(node))
        op = (
            node.name
            if node.name
            else (node.grad_fn.__name__ if node.grad_fn else "leaf")
        )
        label = f"{op}\n{node.data.shape}"
        dot.node(nid, label)
        for parent in node.parents:
            dot.edge(str(id(parent)), nid)
    return dot


# TestPerceptron.test_random_initialization()
# TestPerceptron.test_inputted_weights()
# TestActivations.test_relu()
# TestLinearLayer.test_single_linear_layer()
TestLinearLayer.test_fcn_with_cce(True)
TestLinearLayer.test_fcn_with_mse(True)

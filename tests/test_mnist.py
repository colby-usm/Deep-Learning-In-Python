from data_utils.DataLoader import DataLoader
from data_utils.MNIST import MNIST
from neural.Tensor import Tensor
from neural.LinearLayer import LinearLayer

from neural.MNISTModel import MNISTModel
from neural.Optimizers import SGD

import numpy as np
from tqdm import tqdm

mnist_root = "../datasets/mnist/"

train_data = MNIST(
    mnist_root + "train-images-idx3-ubyte",
    mnist_root + "train-labels-idx1-ubyte",
    "MNIST Train Dataset",
)
test_data = MNIST(
    mnist_root + "t10k-images-idx3-ubyte",
    mnist_root + "t10k-labels-idx1-ubyte",
    "MNIST Test Dataset",
)

train_loader = DataLoader(train_data, name="MNIST Train")
test_loader = DataLoader(test_data, name="MNIST Test")
print(train_loader)
print(test_loader)


model = MNISTModel(
    hl1=LinearLayer(
        (
            Tensor.randn(784, 64),
            Tensor.randn(
                64,
            ),
        ),
        name="hl1",
    ),
    hl2=LinearLayer(
        (
            Tensor.randn(64, 64),
            Tensor.randn(
                64,
            ),
        ),
        name="hl2",
    ),
    o=LinearLayer(
        (
            Tensor.randn(64, 10),
            Tensor.randn(
                10,
            ),
        ),
        name="o",
    ),
)


LR = 1e-5
optimizer = SGD(model.parameters(), lr=LR)
EPOCHS = 5


# train loop
for epoch in tqdm(range(EPOCHS)):
    epoch_loss = 0.0
    n_batches = 0

    for images, labels in train_loader:
        y_pred = model(images)
        loss = y_pred.categorical_cross_entropy(labels)
        graph = loss.backward()
        for node in graph:
            if node.data.shape == (784, 64):
                print(
                    f"graph node id: {id(node)}, grad norm: {np.linalg.norm(node.grad)}"
                )
        print(f"SGD param id: {id(optimizer.params[0])}")
        exit()

    for images, labels in tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        y_pred = model(images)
        loss = y_pred.categorical_cross_entropy(labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.data.item()
        n_batches += 1
    print(f"Epoch {epoch + 1}/{EPOCHS} — loss: {epoch_loss / n_batches:.4f}")

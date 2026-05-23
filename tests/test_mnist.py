from data_utils.DataLoader import DataLoader
from data_utils.MNIST import MNIST
from neural.Tensor import Tensor
from neural.LinearLayer import LinearLayer

from neural.MNISTModel import MNISTModel
from neural.Optimizers import SGD

import numpy as np
import tqdm


LR = 1e-5
EPOCHS = 100
BATCH_SIZE = 1


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

train_loader = DataLoader(train_data, name="MNIST Train", batch_size=BATCH_SIZE)
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


optimizer = SGD(model.parameters(), lr=LR)


for epoch in tqdm.tqdm(range(EPOCHS)):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm.tqdm(train_loader, leave=False):
        optimizer.zero_grad()
        y_pred, logits = model(images)
        loss = logits.softmax_cross_entropy(labels)
        p = list(model.parameters())[0]
        loss.backward()
        optimizer.step()

        epoch_loss += loss.data.item()
        preds = np.argmax(y_pred.data, axis=-1)
        targets = np.argmax(labels, axis=-1)
        correct += np.sum(preds == targets)
        total += len(targets)
    if total > 0:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} — loss: {epoch_loss / total:.4f} — acc: {correct / total:.4f}"
        )

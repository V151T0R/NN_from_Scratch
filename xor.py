import numpy as np

from Functions.Activation_Functions import Tanh
from Functions.Loss_Functions import mse, mse_prime
from Layer.DenseLayer import DenseLayer
from Network.predict import predict
from Network.train import train



x_train = np.reshape(
    [[0, 0], [0, 1], [1, 0], [1, 1]],
    (4, 2, 1),
)
y_train = np.reshape(
    [[0], [1], [1], [0]],
    (4, 1, 1),
)

network = [
    DenseLayer(2, 3),
    Tanh(),
    DenseLayer(3, 1),
    Tanh(),
]

train(
    network,
    mse,
    mse_prime,
    x_train,
    y_train,
    epochs=10000,
    learning_rate=0.1,
)

print("\nPredictions:")
for x, y in zip(x_train, y_train):
    output = predict(network, x)
    print(f"input={x.ravel().astype(int)}, predicted={output.item():.4f}, target={y.item():.0f}")


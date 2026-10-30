# Neural Network from Scratch (`NN_from_Scratch`)

A clean, modular implementation of a feedforward neural network built from scratch in Python using only **NumPy**. This project is designed to make the core mechanics and mathematics of deep learning—forward propagation, activation functions, loss computation, backpropagation, and gradient descent—transparent and easy to follow.

---

## Key Features

- **Pure NumPy Implementation**: No deep learning frameworks (PyTorch, TensorFlow, Keras)—all matrix operations and gradients are explicitly derived and implemented.
- **Modular Layer Architecture**: Every component (`DenseLayer`, `Activation`, `Softmax`, `LossLayer`) inherits from a unified abstract base class (`Layer`) implementing `forward()` and `backward()` methods.
- **Built-in Activations & Losses**:
  - **Activations**: `Tanh`, `Sigmoid`, and `Softmax`
  - **Losses**: Mean Squared Error (`MSE`) and `BinaryCrossEntropy` wrapped as reusable `LossLayer` classes
- **End-to-End Training & Inference Pipeline**: Simple `train()` and `predict()` utilities with per-epoch loss tracking.
- **Working XOR Benchmark**: Demonstrates non-linear decision boundary learning on the classic XOR problem in `xor.py`.

---

## Project Structure

```text
NN/
├── Functions/
│   ├── __init__.py                # Exports Sigmoid, Tanh, MSE, BinaryCrossEntropy
│   ├── Activation_Functions.py    # Concrete activation layers (Tanh, Sigmoid)
│   └── Loss_Functions.py          # Concrete loss layers (MSE, BinaryCrossEntropy)
├── Layer/
│   ├── __init__.py                # Exports Layer, DenseLayer, Activation, LossLayer, Softmax
│   ├── BaseLayer.py               # Abstract base class defining forward() and backward()
│   ├── DenseLayer.py              # Fully connected linear layer with weights and biases
│   ├── ActivationLayer.py         # Element-wise activation wrapper layer
│   ├── LossLayer.py               # Loss computation and initial gradient wrapper layer
│   └── SoftmaxLayer.py            # Softmax activation layer with Jacobian backward pass
├── Network/
│   ├── __init__.py                # Exports predict and train
│   ├── predict.py                 # Sequential forward pass across network layers
│   └── train.py                   # Stochastic gradient descent training loop
├── xor.py                         # Executable example solving the XOR problem
└── NN_From_Scratch_Lecture.md     # Step-by-step companion lecture notes
```

---

## How It Works & Mathematical Foundations

### 1. Shape Convention
Each single sample is represented as a **column vector** of shape `(features, 1)`.
- For an input $\mathbf{x} \in \mathbb{R}^{n \times 1}$, a `DenseLayer(n, m)` stores:
  - Weights $\mathbf{W} \in \mathbb{R}^{m \times n}$
  - Bias $\mathbf{b} \in \mathbb{R}^{m \times 1}$

### 2. Layer Contract (`Layer/BaseLayer.py`)
All layers inherit from the abstract `Layer` class and implement two methods:
- `forward(input)`: Computes the layer's output from the given input (caching values needed for backpropagation).
- `backward(output_gradient, learning_rate)`: Receives $\frac{\partial L}{\partial \mathbf{Y}}$ from the subsequent layer, updates internal parameters if applicable, and returns $\frac{\partial L}{\partial \mathbf{X}}$ to propagate backward.

### 3. Dense Layer (`Layer/DenseLayer.py`)
- **Forward Pass**:
  $$\mathbf{Y} = \mathbf{W}\mathbf{X} + \mathbf{b}$$
- **Backward Pass**: Given upstream gradient $\frac{\partial L}{\partial \mathbf{Y}}$:
  $$\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{X}^T, \quad \frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{Y}}, \quad \frac{\partial L}{\partial \mathbf{X}} = \mathbf{W}^T \frac{\partial L}{\partial \mathbf{Y}}$$
- **Parameter Update (Gradient Descent)**:
  $$\mathbf{W} \leftarrow \mathbf{W} - \eta \frac{\partial L}{\partial \mathbf{W}}, \quad \mathbf{b} \leftarrow \mathbf{b} - \eta \frac{\partial L}{\partial \mathbf{b}}$$

### 4. Activation Layers (`Layer/ActivationLayer.py`, `Functions/Activation_Functions.py`, `Layer/SoftmaxLayer.py`)
- **Element-wise Activations (`Tanh`, `Sigmoid`)**:
  - **Forward Pass**: $\mathbf{Y} = f(\mathbf{X})$
  - **Backward Pass**: $\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \odot f'(\mathbf{X})$
  - **Tanh**: $f(x) = \tanh(x)$, $f'(x) = 1 - \tanh^2(x)$
  - **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$, $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- **Softmax (`Softmax`)**:
  - **Forward Pass**: $y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$
  - **Backward Pass**: Multiplies the Softmax Jacobian $(I - \mathbf{Y}^T) \odot \mathbf{Y}$ with the upstream output gradient.

### 5. Loss Layers (`Layer/LossLayer.py`, `Functions/Loss_Functions.py`)
`LossLayer` encapsulates both the scalar loss calculation (`forward(input, target)`) and the initial gradient $\frac{\partial L}{\partial \hat{\mathbf{Y}}}$ (`backward()`):
- **Mean Squared Error (`MSE`)**:
  $$L(\mathbf{y}, \hat{\mathbf{y}}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2, \quad \frac{\partial L}{\partial \hat{\mathbf{y}}} = \frac{2}{N}(\hat{\mathbf{y}} - \mathbf{y})$$
- **Binary Cross-Entropy (`BinaryCrossEntropy`)**:
  $$L(\mathbf{y}, \hat{\mathbf{y}}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$
  $$\frac{\partial L}{\partial \hat{\mathbf{y}}} = \frac{1}{N} \left( \frac{1 - \mathbf{y}}{1 - \hat{\mathbf{y}}} - \frac{\mathbf{y}}{\hat{\mathbf{y}}} \right)$$

### 6. Training Loop (`Network/train.py` & `Network/predict.py`)
For each epoch and each training sample $(\mathbf{x}, \mathbf{y})$:
1. **Forward Propagation**: Pass $\mathbf{x}$ sequentially through every layer in `network` via `predict(network, x)`.
2. **Loss Evaluation**: Compute sample loss via `loss_layer.forward(output, y)`.
3. **Backward Propagation**: Obtain initial gradient via `loss_layer.backward()` and propagate it through `reversed(network)`, updating weights and biases at each learnable layer.

---

## Getting Started

### Prerequisites
- Python 3.8+
- [NumPy](https://numpy.org/)

Install NumPy if you haven't already:
```bash
pip install numpy
```

### Running the XOR Example

Clone the repository, navigate to the `NN` directory, and run `xor.py`:

```bash
cd NN
python xor.py
```

#### Example Usage (`xor.py`)
```python
import numpy as np
from Functions.Activation_Functions import Tanh
from Functions.Loss_Functions import MSE
from Layer.DenseLayer import DenseLayer
from Network.predict import predict
from Network.train import train

# Prepare XOR dataset: 4 samples, each of shape (2, 1)
x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Define network architecture
network = [
    DenseLayer(2, 3),
    Tanh(),
    DenseLayer(3, 1),
    Tanh(),
]

# Initialize MSE loss and train
loss = MSE()
train(
    network,
    loss.loss,
    loss.loss_prime,
    x_train,
    y_train,
    epochs=10000,
    learning_rate=0.1,
)

# Run inference
for x, y in zip(x_train, y_train):
    output = predict(network, x)
    print(f"input={x.ravel().astype(int)}, predicted={output.item():.4f}, target={y.item():.0f}")
```

#### Expected Output
```text
Training started: samples=4, epochs=10000, learning_rate=0.1
Epoch 1/10000 - loss: 0.523412
...
Epoch 10000/10000 - loss: 0.000024
Training complete: final_loss=0.000024

Predictions:
input=[0 0], predicted=0.0006, target=0
input=[0 1], predicted=0.9949, target=1
input=[1 0], predicted=0.9949, target=1
input=[1 1], predicted=0.0012, target=0
```

---

## Extending the Network

Because all layers follow the `Layer` interface in `Layer/BaseLayer.py`, adding custom layers or activations takes only a few lines of code:

### Adding a Custom Activation Function (e.g., ReLU)
```python
import numpy as np
from Layer.ActivationLayer import Activation

class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)
```

---




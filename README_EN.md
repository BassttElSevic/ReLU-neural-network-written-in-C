[简体中文](README.md) | [English](README_EN.md)

# C-ReLU Shape Classifier

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&lines=Hello+there!;Welcome+to+my+project)](https://git.io/typing-svg)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Repository](https://img.shields.io/badge/GitHub-Repo-181717?style=flat-square&logo=github)](https://github.com/BassttElSevic/ReLU-neural-network-written-in-C/)
[![UwU](https://img.shields.io/badge/-UwU-ff69b4?style=flat-square)](https://github.com/BassttElSevic/Linear-Algebra-Operators-and-Their-Decompositions/)
[![C](https://img.shields.io/badge/C-00599C?style=flat-square&logo=c&logoColor=white)](https://www.iso.org/standard/74528.html)
[![ReLU](https://img.shields.io/badge/ReLU-activation-7B2FBE?style=flat-square&logo=brain&logoColor=white)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))

(*BTW*, I trained it on [![ThinkPad](https://img.shields.io/badge/ThinkPad-EE2624?style=flat-square&logo=thinkpad&logoColor=white)](https://www.lenovo.com/us/en/thinkpad) and my cpu is [![Intel](https://img.shields.io/badge/Intel-0071C5?style=flat-square&logo=intel&logoColor=white)](https://www.intel.com/) and my OS is [![Linux](https://img.shields.io/badge/Linux-000000?style=flat-square&logo=linux&logoColor=white)](https://www.linux.org/)

[![GCC](https://img.shields.io/badge/GCC-FF6600?style=flat-square&logo=gcc&logoColor=white)](https://gcc.gnu.org/)
[![CMake](https://img.shields.io/badge/CMake-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org/)

A neural network written in pure C that distinguishes rectangles from circles. The input is a grayscale image of 50x50 (pixel values 0.0 to 1.0), and the output is two classes: rectangle (index 0) and circle (index 1).

The model starts from random weights and learns automatically through backpropagation and gradient descent with momentum. The training data is not manually labeled; it is generated on the fly by the program as random rectangles and circles.

## Build

You need gcc and CMake. CMake automatically enables OpenMP to use all of your CPU cores.

```bash
cmake -S . -B build
cmake --build build -j
```

The resulting program is `build/shape_classifier`.

## Run

```bash
# Train and save the model to data/shape_net.bin
./build/shape_classifier train 12345

# Load the saved model and predict directly (no retraining)
./build/shape_classifier predict
```

The number after `train` is an optional random seed. A different seed gives a slightly different result.

After training, the following files are generated:

- `data/shape_net.bin`: the trained model weights.
- `data/training_curve.csv`: loss and accuracy per epoch.
- `data/sample_epoch_*.ppm`: sample images generated during training.

## Visualize the training curve

To see the loss curve, use the script `tools/plot_curve.py`. It reads `data/training_curve.csv` and plots the training loss plus the training/validation accuracy with matplotlib.

```bash
python3 tools/plot_curve.py
```

The result is written to `data/training_curve.png`. It requires python3 and matplotlib. A Chinese font (such as WenQuanYi Micro Hei) must be installed on the system; the script finds it automatically.

## Network architecture

```text
Input 2500 pixels (50x50)
   ↓                          fully connected layer 1, 128 units, ReLU
Hidden layer 1: 128
   ↓                          fully connected layer 2, 64 units, ReLU
Hidden layer 2: 64
   ↓                          fully connected layer 3, 2 units, softmax
Output: 2 classes (rectangle / circle)
```

Training uses cross-entropy loss and a gradient-descent optimizer with momentum (momentum 0.9, learning rate 0.02). The gradient computation is parallelized with OpenMP, distributing a batch of samples across different threads.

## About multi-layer networks

A network with only an input layer and an output layer, with no hidden layer in between, is called a single-layer perceptron. It can only solve linearly separable problems: what it learns is a linear combination between input and output, equivalent to drawing a line in the input space to separate the two classes. The pixel distributions of rectangles and circles are not linearly separable, so a single layer is not enough.

A multi-layer network inserts one or more hidden layers between the input and the output. Each layer takes its input, applies a linear transformation (multiply by weights, add bias), and then passes the result through a nonlinear activation function. The point of stacking multiple layers is that the lower layers can learn low-level features, and the higher layers combine these outputs to express more complex functions. The more hidden layers and units, the more complex the function that can be approximated, but training is also slower and more prone to overfitting.

This project uses two hidden layers (three weight layers in total). Each layer first does `weight * input + bias`, then passes through ReLU.

## About ReLU

The activation function adds a nonlinearity after the linear transformation; otherwise, no matter how many layers are stacked, the network is still mathematically a linear function and is no different from a single layer. The function you are familiar with that bounds the output between 0 and 1 is usually sigmoid:

```text
sigmoid(x) = 1 / (1 + e^(-x)), result in (0, 1)
```

The problem with sigmoid is that when x is very large or very small, the function changes very slowly and its gradient is close to 0, which makes the weights hard to update (this is called the vanishing gradient problem). This is especially bad for deeper networks.

ReLU (Rectified Linear Unit) is another activation function, defined very simply:

```text
relu(x) = max(0, x)
```

That is, if the input is greater than 0 it is passed through unchanged, and if it is less than or equal to 0 it outputs 0. Its advantages are:

- Extremely fast to compute, no exponential operations.
- In the interval where the input is greater than 0, the gradient is always 1, so it does not suffer the vanishing gradient problem of sigmoid, making it more stable for training deeper, faster networks.
- The output has no upper bound, so a larger value can express a stronger feature response.

The cost is: in the interval where the input is less than or equal to 0, the gradient is 0, which stops that neuron from learning in that direction. This is mitigated in practice by proper initialization and a smaller learning rate.

In the code, ReLU appears after each layer:

```c
// layer 1: linear transform, then ReLU
h1[j] = fmaxf(0.0f, net->b1[j] + dot(INPUT_SIZE, &net->w1[j * INPUT_SIZE], input));
```

The output layer is different. Because we want the probability of each class (rectangle and circle), and probabilities range from 0 to 1 and sum to 1, the output layer uses softmax instead of ReLU:

```text
softmax(z)[k] = e^(z[k]) / Σ_j e^(z[j])
```

ReLU is only used in the hidden layers; softmax is only used in the output layer.

## Directory structure

```text
include/       header declarations
  layer.h      layer definition, drawing, saving
  network.h    network structure, training, prediction, save/load
  dataset.h    random sample generation
  rng.h        random number wrapper
src/           implementation
  main.c       train/predict entry point
  layer.c      rectangle, circle drawing, PPM/binary save
  network.c    forward, backward, momentum update, model read/write
  dataset.c    generate random rectangle or circle samples
  rng.c        random numbers
tools/         helper scripts
  plot_curve.py   training curve visualization
data/          training artifacts (ignored by git)
```

## License

MIT License, copyright by <img width="500" height="500" alt="图片" src="https://github.com/user-attachments/assets/c5a5ea84-8b31-4641-81b1-cea08a739ba7" /> Basstt ElSevic. See [LICENSE](LICENSE).

## Choose life

<img width="1080" height="723" alt="ChooseLife" src="https://github.com/user-attachments/assets/c5b28f73-074d-4b5d-9498-a77cd4d51561" />

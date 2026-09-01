# C-ReLU Shape Classifier

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&lines=Hello+there!;Welcome+to+my+project)](https://git.io/typing-svg)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Repository](https://img.shields.io/badge/GitHub-Repo-181717?style=flat-square&logo=github)](https://github.com/BassttElSevic/ReLU-neural-network-written-in-C/)
[![UwU](https://img.shields.io/badge/-UwU-ff69b4?style=flat-square)](https://github.com/BassttElSevic/Linear-Algebra-Operators-and-Their-Decompositions/)
[![C](https://img.shields.io/badge/C-00599C?style=flat-square&logo=c&logoColor=white)](https://www.iso.org/standard/74528.html)
[![ReLU](https://img.shields.io/badge/ReLU-激活函数-7B2FBE?style=flat-square&logo=brain&logoColor=white)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))

(*BTW*,I trained it on [![ThinkPad](https://img.shields.io/badge/ThinkPad-EE2624?style=flat-square&logo=thinkpad&logoColor=white)](https://www.lenovo.com/us/en/thinkpad) and my cpu is [![Intel](https://img.shields.io/badge/Intel-0071C5?style=flat-square&logo=intel&logoColor=white)](https://www.intel.com/) and my OS is [![Linux](https://img.shields.io/badge/Linux-000000?style=flat-square&logo=linux&logoColor=white)](https://www.linux.org/)

[![GCC](https://img.shields.io/badge/GCC-FF6600?style=flat-square&logo=gcc&logoColor=white)](https://gcc.gnu.org/)
[![CMake](https://img.shields.io/badge/CMake-064F8C?style=flat-square&logo=cmake&logoColor=white)](https://cmake.org/)

一个用纯 C 写的神经网络，用来区分矩形和圆形。输入是一张 50x50 的灰度图（像素值 0.0 到 1.0），输出两个类别：矩形（索引 0）和圆形（索引 1）。

模型从随机权重开始，通过反向传播和带有动量的梯度下降自动学习。训练数据不是手工标注的，而是程序实时生成的随机矩形和圆形。

## 编译

需要 gcc 和 CMake。CMake 会自动启用 OpenMP 来使用 CPU 多核并行。

```bash
cmake -S . -B build
cmake --build build -j
```

生成的程序在 `build/shape_classifier`。

## 运行

```bash
# 训练并保存模型到 data/shape_net.bin
./build/shape_classifier train 12345

# 加载已保存的模型直接预测（不重新训练）
./build/shape_classifier predict
```

`train` 后面的数字是可选的随机种子。改用其它种子会得到略不同的结果。

训练结束后会生成：

- `data/shape_net.bin`：训练好的模型权重。
- `data/training_curve.csv`：每轮的损失和准确率。
- `data/sample_epoch_*.ppm`：训练过程中随机生成的样例图。

## 可视化训练曲线

要看到损失下降的曲线，用脚本 `tools/plot_curve.py`。它读取 `data/training_curve.csv`，用 matplotlib 画出训练损失和训练/验证准确率两条曲线。

```bash
python3 tools/plot_curve.py
```

结果输出到 `data/training_curve.png`。需要 python3 和 matplotlib。系统需装有中文字体（如文泉驿微米黑），脚本会自动查找。

## 网络结构

```text
输入 2500 像素（50x50）
   ↓                          全连接层 1，128 个单元，ReLU
隐藏层 1：128
   ↓                          全连接层 2，64 个单元，ReLU
隐藏层 2：64
   ↓                          全连接层 3，2 个单元，softmax
输出：2 类（矩形 / 圆形）
```

训练用的是交叉熵损失，优化器是带动量的梯度下降（momentum 0.9，学习率 0.02）。梯度计算部分用 OpenMP 并行，把一批样本分给不同线程。

## 关于多层网络

一个只含输入层和输出层的网络，中间没有隐藏层，叫做单层感知机。它只能解决线性可分的问题：它学到的是一张输入和输出之间的线性组合，相当于在输入空间里画一条直线来分隔两类。矩形和圆形的像素分布并非线性可分，所以单层不够。

多层网络在输入和输出之间插入一层或多层隐藏层。每一层都把自己的输入做一次线性变换（乘权重、加偏置），再经过一个非线性激活函数。堆叠多层的意义是：低层可以学到底层特征，高层在底层的输出上继续组合，从而表达更复杂的函数关系。隐藏层越多、单元越多，能逼近的函数就越复杂，但训练也更慢、更容易过拟合。

本项目用的是两层隐藏层（共三层权重）。每层先做 `weight * input + bias`，再经过 ReLU。

## 关于 ReLU

激活函数的作用是在线性变换之后加一个非线性，否则多层网络无论叠多少层，数学上都仍然是一个线性函数，和单层没有区别。你熟悉的、把输出限制在 0 到 1 之间的函数通常是 sigmoid：

```text
sigmoid(x) = 1 / (1 + e^(-x))，结果在 (0, 1)
```

sigmoid 的问题是：当 x 很大或很小时，函数变化很慢，对应的梯度接近于 0，导致权重很难更新（称为梯度消失）。这对深层的网络尤其不利。

ReLU（Rectified Linear Unit，线性整流单元）是另一种激活函数，定义非常简单：

```text
relu(x) = max(0, x)
```

也就是：输入大于 0 就原样输出，输入小于等于 0 就输出 0。它的优势是：

- 计算极快，没有指数运算。
- 输入大于 0 的区间梯度恒为 1，不会像 sigmoid 那样梯度消失，训练更深、更快的网络更稳定。
- 输出不加上限，可以用更大的值表达更强的特征响应。

代价是：输入小于等于 0 的区间梯度为 0，这会让神经元在该方向上停止学习。实践中通过合理初始化和较小的学习率缓解。

在代码里，ReLU 出现在每一层之后：

```c
// 层1: 先做线性变换,再 ReLU
h1[j] = fmaxf(0.0f, net->b1[j] + dot(INPUT_SIZE, &net->w1[j * INPUT_SIZE], input));
```

输出层不同。因为我们要的是"矩形和圆形各自的概率"，概率在 0 到 1 之间且和为 1，所以输出层用 softmax，而不是 ReLU：

```text
softmax(z)[k] = e^(z[k]) / Σ_j e^(z[j])
```

ReLU 只用在隐藏层，softmax 只用在输出层。

## 目录结构

```text
include/       头文件声明
  layer.h      图层定义和填充、保存
  network.h    网络结构、训练、预测、保存/加载
  dataset.h    随机样本生成
  rng.h        随机数包装
src/           实现
  main.c       训练/预测入口
  layer.c      矩形、圆形绘制，PPM/二进制保存
  network.c    前向、反向、动量更新、模型读写
  dataset.c    随机生成矩形或圆形样本
  rng.c        随机数
tools/         辅助脚本
  plot_curve.py  训练曲线可视化
data/          训练产物（被 git 忽略）
```

## License

MIT License，版权归 Basstt ElSevic 所有。详见 [LICENSE](LICENSE)。

## Choose life
<img width="1080" height="723" alt="ChooseLife" src="https://github.com/user-attachments/assets/c5b28f73-074d-4b5d-9498-a77cd4d51561" />

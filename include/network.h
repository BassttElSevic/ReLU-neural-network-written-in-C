#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

/* =====================================================================
 * 神经网络模块:前向传播(预测)、反向传播(训练)、模型持久化。
 * ===================================================================== */

/* 输入为把一张 WIDTH x HEIGHT 图层拉平后的像素(0.0 背景 / 1.0 前景)。
 * 网络结构: 输入 -> 全连接(ReLU) -> 全连接(ReLU) -> 全连接(softmax)。
 * 输出 2 类: 索引 0 = 矩形, 索引 1 = 圆形。 */

/* 各层单元数(也决定了权重矩阵的形状):
 *   输入层:50x50 = 2500 个像素,展平成一维
 *   隐藏层1:128 个单元
 *   隐藏层2:64 个单元
 *   输出层:2 个类别(矩形/圆形)
 * */
#define INPUT_SIZE   (WIDTH * HEIGHT)   /* 2500 */
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define OUTPUT_SIZE  2

/* 网络结构体,里面存的就是"学到的所有参数":
 *   w1,b1  第一层权重(A x B)和偏置(B)
 *   w2,b2  第二层权重和偏置
 *   w3,b3  输出层权重和偏置
 *
 * 每个权重/偏置还配了一个同尺寸的 "v" 数组(如 vw1, vb1),
 * 这是动量优化器的"速度缓冲",记录上次更新的方向和幅度。
 * (详见 network.c 的 network_update) */
typedef struct {
    float *w1, *b1;   /* w1: HIDDEN1 x INPUT,  b1: HIDDEN1 */
    float *w2, *b2;   /* w2: HIDDEN2 x HIDDEN1, b2: HIDDEN2 */
    float *w3, *b3;   /* w3: OUTPUT  x HIDDEN2, b3: OUTPUT  */

    float *vw1, *vb1;   /* 动量速度缓冲 */
    float *vw2, *vb2;
    float *vw3, *vb3;
} Network;

/* 梯度累加器:尺寸和上面的权重/偏置一一对应。
 * 训练时把一批样本的梯度累加到这里,再统一用它更新权重。 */
typedef struct {
    float *gw1, *gb1;
    float *gw2, *gb2;
    float *gw3, *gb3;
} Gradients;

/* 创建网络:分配内存并随机初始化权重 */
Network   *network_create(void);
/* 释放网络占用的内存 */
void       network_free(Network *net);

/* 创建梯度累加器 */
Gradients *gradients_create(void);
/* 释放梯度累加器 */
void       gradients_free(Gradients *g);

/* 前向传播:输入一张图(input,2500),把各类别概率写入 probs(probs[0]+probs[1]=1) */
void       network_forward(Network *net, const float *input, float *probs);

/* 并行累加一批样本(count 个)的梯度到 g,返回该批平均交叉熵损失。
 * 用 OpenMP 多线程,多个线程各自处理不同样本。 */
float network_accumulate_batch(Network *net, Gradients *g,
                               const float *input, const int *labels,
                               int count);

/* 用累积梯度做一次带动量(平均梯度)的同步更新 */
void  network_update(Network *net, Gradients *g, float lr, float momentum, int count);

/* 预测:返回 1=圆形, 0=矩形 */
int   network_predict(Network *net, const float *input);

/* 保存模型权重到文件(成功 0,失败 -1) */
int      network_save(const Network *net, const char *file_path);
/* 从文件加载模型权重(失败返回 NULL) */
Network *network_load(const char *file_path);

#endif /* NETWORK_H */

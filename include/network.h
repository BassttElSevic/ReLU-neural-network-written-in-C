#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

/* 输入为把一张 WIDTH x HEIGHT 图层拉平后的像素(0.0 背景 / 1.0 前景)。
 * 网络结构: 输入 -> 全连接(ReLU) -> 全连接(ReLU) -> 全连接(softmax)。
 * 输出 2 类: 索引 0 = 矩形, 索引 1 = 圆形。
 */

#define INPUT_SIZE   (WIDTH * HEIGHT)
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64
#define OUTPUT_SIZE  2

typedef struct {
    float *w1, *b1;   /* w1: HIDDEN1 x INPUT,  b1: HIDDEN1 */
    float *w2, *b2;   /* w2: HIDDEN2 x HIDDEN1, b2: HIDDEN2 */
    float *w3, *b3;   /* w3: OUTPUT  x HIDDEN2, b3: OUTPUT  */

    /* 动量优化器用到的速度缓冲,尺寸与对应权重/偏置一致 */
    float *vw1, *vb1;
    float *vw2, *vb2;
    float *vw3, *vb3;
} Network;

/* 梯度累积器: 尺寸与对应权重/偏置一致,用于批量并行累加梯度 */
typedef struct {
    float *gw1, *gb1;
    float *gw2, *gb2;
    float *gw3, *gb3;
} Gradients;

Network   *network_create(void);
void       network_free(Network *net);

Gradients *gradients_create(void);
void       gradients_free(Gradients *g);

/* forward: 输入 input(INPUT_SIZE),把各类别概率写入 probs(OUTPUT_SIZE)。 */
void       network_forward(Network *net, const float *input, float *probs);

/* 并行累加一批样本(count 个)的梯度到 g,返回该批平均交叉熵损失。
 * 输入: input 为 count*INPUT_SIZE 的扁平数组,labels 为 count 个标签。
 * 该函数使用 OpenMP 并行,多个线程各自处理样本并写入私有梯度缓冲。
 */
float network_accumulate_batch(Network *net, Gradients *g,
                               const float *input, const int *labels,
                               int count);

/* 用累积梯度做一次带动量(均值梯度)的同步更新。count 为这批样本数。 */
void  network_update(Network *net, Gradients *g, float lr, float momentum, int count);

int   network_predict(Network *net, const float *input);

/* 模型权重保存/加载(二进制)。 */
int      network_save(const Network *net, const char *file_path);
Network *network_load(const char *file_path);

#endif /* NETWORK_H */

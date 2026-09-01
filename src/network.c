#include "network.h"
#include "rng.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/* =====================================================================
 * 这个文件是神经网络的核心。
 *
 * 神经网络在干什么?一句话:给一堆输入,算出每个类别的概率,然后根据
 * "算错了多少"去修正内部的一堆数字(权重),让下次算得更准。
 *
 * 文件里每一段对应一件事:
 *   - init_weights    -> 初始化权重(训练前的起点)
 *   - network_forward -> 前向传播(用当前权重做一次预测)
 *   - accumulate_one  -> 对一个样本算"梯度"(该往哪个方向改权重)
 *   - network_update  -> 用梯度更新权重(带动量,让学习更快更稳)
 *   - save/load       -> 把学到的权重存硬盘,下次直接读,不用重训
 * ===================================================================== */

/* ------------- 工具函数 ------------- */

/* 申请 n 个 float 的内存,并全部清零。
 * calloc 是 "call + zero":分配内存的同时把所有字节置 0。
 * 对比 malloc:malloc 只分配不清零,里面的值是随机的垃圾值。
 * 这里用 calloc,保证初始的偏置、动量速度缓冲都是 0。 */
static float *alloc(int n) { return (float *)calloc((size_t)n, sizeof(float)); }

/* 向量点积: a 和 b 是两个等长的数组,逐个相乘再累加。
 * 写成 sum += a[i] * b[i]。
 *
 * 这个函数是"全连接层"的本质。一个神经元做的事就是:
 *      把每个输入乘上对应权重,然后全部加起来(再加偏置)。
 * 就是点积。所以点积 = 一个神经元的一次"线性变换"。 */
static float dot(int n, const float *a, const float *b)
{
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* 用均匀分布初始化权重,范围 [-gain, gain],gain 取 1/sqrt(fan_in)。
 *
 * 为什么要随机初始化?如果所有权重都设成 0,那所有神经元算出来都一样,
 * 训练时它们也会得到一样的梯度,永远无法"分化",网络等于废了。
 * 所以要用小随机数打破对称。
 *
 * 为什么范围取 1/sqrt(fan_in)?
 * fan_in 是"这个神经元有多少个输入"。比如输入层有 2500 个像素,
 * 每个像素的权重不能太大,否则 2500 个加起来可能爆炸(数值都是 0~1 的像素,
 * 乘个大权重再累加 2500 次,结果巨大)。除以 sqrt(fan_in) 是一种经典的
 * "保持方差稳定"的初始化技巧,让每层输出的数量级大致相同,训练更稳。
 *
 * rand()/RAND_MAX*2-1 把随机数映射到 [-1, 1]:
 *   rand()/RAND_MAX -> [0, 1]
 *   *2 - 1          -> [-1, 1]
 * 再乘 gain 缩放到想要的量级。 */
static void init_weights(float *w, int fan_in, int fan_out)
{
    float gain = 1.0f / sqrtf((float)fan_in);
    for (int i = 0; i < fan_in * fan_out; ++i)
        w[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * gain;
}

/* ------------- 生命周期: 创建/释放 网络 和 梯度缓冲 ------------- */

/* 创建一个网络,并分配所有权重/偏置/动量缓冲的存储。
 *
 * 网络有 3 层权重(两组隐藏层 + 一组输出层),每层都有:
 *   - 权重 w   : 和上一层连接起来的"线"的粗细,数量 = 本层单元数 x 上一层输出数
 *   - 偏置 b   : 每个神经元自带的"基础值",数量 = 本层单元数
 *
 * 具体形状(见 network.h 的宏):
 *   w1: HIDDEN1_SIZE x INPUT_SIZE   -> 128 x 2500  (输入层连到隐藏层1)
 *   b1: HIDDEN1_SIZE                -> 128
 *   w2: HIDDEN2_SIZE x HIDDEN1_SIZE -> 64 x 128    (隐藏层1连到隐藏层2)
 *   b2: HIDDEN2_SIZE                -> 64
 *   w3: OUTPUT_SIZE x HIDDEN2_SIZE  -> 2 x 64      (隐藏层2连到输出层)
 *   b3: OUTPUT_SIZE                 -> 2
 *
 * 另外给每个权重/偏置配了一个同尺寸的 "v" 数组(vw1, vb1...),
 * 这是动量优化器的"速度缓冲",存上一次更新的方向,后面 network_update 会用。
 * */
Network *network_create(void)
{
    Network *net = (Network *)calloc(1, sizeof(Network));
    if (!net) return NULL;

    net->w1 = alloc(HIDDEN1_SIZE * INPUT_SIZE);
    net->b1 = alloc(HIDDEN1_SIZE);
    net->w2 = alloc(HIDDEN2_SIZE * HIDDEN1_SIZE);
    net->b2 = alloc(HIDDEN2_SIZE);
    net->w3 = alloc(OUTPUT_SIZE * HIDDEN2_SIZE);
    net->b3 = alloc(OUTPUT_SIZE);

    /* 动量速度缓冲,尺寸和对应权重/偏置一一对应 */
    net->vw1 = alloc(HIDDEN1_SIZE * INPUT_SIZE);
    net->vb1 = alloc(HIDDEN1_SIZE);
    net->vw2 = alloc(HIDDEN2_SIZE * HIDDEN1_SIZE);
    net->vb2 = alloc(HIDDEN2_SIZE);
    net->vw3 = alloc(OUTPUT_SIZE * HIDDEN2_SIZE);
    net->vb3 = alloc(OUTPUT_SIZE);

    /* 随机初始化三组权重(偏置保持 0 即可) */
    init_weights(net->w1, INPUT_SIZE, HIDDEN1_SIZE);
    init_weights(net->w2, HIDDEN1_SIZE, HIDDEN2_SIZE);
    init_weights(net->w3, HIDDEN2_SIZE, OUTPUT_SIZE);

    return net;
}

/* 释放网络占用的所有内存。
 * free 必须和 malloc/calloc 成对使用,否则会造成"内存泄漏"(程序占着内存不放)。
 * */
void network_free(Network *net)
{
    if (!net) return;
    free(net->w1); free(net->b1);
    free(net->w2); free(net->b2);
    free(net->w3); free(net->b3);
    free(net->vw1); free(net->vb1);
    free(net->vw2); free(net->vb2);
    free(net->vw3); free(net->vb3);
    free(net);
}

/* 创建梯度缓冲。
 *
 * 什么是"梯度"?
 * 梯度 = 每个权重该往哪个方向改、改多少,才能让"损失"(算错的程度)变小。
 * 打个比方:你在山上想找最低点(损失最小),梯度就是你脚下每个方向坡有多陡。
 * 你要朝下坡走。梯度就是"坡"这个信息。
 *
 * 我们不是每学一个样本就立刻改权重,而是先积累一批样本的梯度(记在 g 里),
 * 积累完再统一改一次。这样更稳(后面 network_accumulate_batch 会说明)。
 * 所以 g 就是"这批样本梯度的累加器"。
 * */
Gradients *gradients_create(void)
{
    Gradients *g = (Gradients *)calloc(1, sizeof(Gradients));
    if (!g) return NULL;
    g->gw1 = alloc(HIDDEN1_SIZE * INPUT_SIZE);
    g->gb1 = alloc(HIDDEN1_SIZE);
    g->gw2 = alloc(HIDDEN2_SIZE * HIDDEN1_SIZE);
    g->gb2 = alloc(HIDDEN2_SIZE);
    g->gw3 = alloc(OUTPUT_SIZE * HIDDEN2_SIZE);
    g->gb3 = alloc(OUTPUT_SIZE);
    return g;
}

void gradients_free(Gradients *g)
{
    if (!g) return;
    free(g->gw1); free(g->gb1);
    free(g->gw2); free(g->gb2);
    free(g->gw3); free(g->gb3);
    free(g);
}

/* ------------- 前向传播 ------------- */

/* 前向传播:用当前权重,对一张输入图算出每个类别的概率。
 *
 * 输入 input:长度为 2500 的数组(50x50 像素拉平),像素值为 0 或 1。
 * 输出 probs:长度为 2 的数组,probs[0] 是"矩形"的概率,probs[1] 是"圆形"的概率,
 *            两者相加 = 1。
 *
 * 数据如何一层层流过去(这就是"多层网络"):
 *
 *   输入(2500) -> 隐藏层1(128, ReLU) -> 隐藏层2(64, ReLU) -> 输出(2, softmax)
 *
 * 每一步都是:先做线性变换(点积+偏置),再过一个激活函数。
 * 线性变换 =  dot(上一层输出, 本层权重) + 偏置
 * 激活函数 =  ReLU(max(0, x)) 或 softmax
 * */
void network_forward(Network *net, const float *input, float *probs)
{
    /* 用栈上局部数组缓存各层的输出。h1 是隐藏层1的输出,h2 是隐藏层2的输出。 */
    float h1[HIDDEN1_SIZE], h2[HIDDEN2_SIZE], logits[OUTPUT_SIZE];

    /* 层1: 每个隐藏单元 z = 偏置 + 权重与输入的点积,再 ReLU。
     * &net->w1[j * INPUT_SIZE] 是取 w1 数组里第 j 个神经元的权重行
     * (w1 是扁平的,第 j 个神经元对应从下标 j*INPUT_SIZE 开始的连续 INPUT_SIZE 个数)。
     * dot(...) 把这份权重跟输入逐项相乘相加 = 该神经元的"原始分数"。
     * fmaxf(0.0f, z) 就是 ReLU:z 小于 0 截成 0,大于 0 保留。
     * */
    for (int j = 0; j < HIDDEN1_SIZE; ++j)
        h1[j] = fmaxf(0.0f, net->b1[j] + dot(INPUT_SIZE, &net->w1[j * INPUT_SIZE], input));

    /* 层2: 同样的过程,但输入变成了上一层的输出 h1,维度是 HIDDEN1_SIZE(128)。 */
    for (int j = 0; j < HIDDEN2_SIZE; ++j)
        h2[j] = fmaxf(0.0f, net->b2[j] + dot(HIDDEN1_SIZE, &net->w2[j * HIDDEN1_SIZE], h1));

    /* 输出层:算出两个"原始分数"(logits)。
     * logits 还没变成概率,它只是每个类别的得分,可以是任意实数。 */
    for (int k = 0; k < OUTPUT_SIZE; ++k)
        logits[k] = net->b3[k] + dot(HIDDEN2_SIZE, &net->w3[k * HIDDEN2_SIZE], h2);

    /* ------ softmax:把 logits 变成和为 1 的概率 ------
     * softmax(z)[k] = e^(z[k]) / Σ_j e^(z[j])
     *
     * 为什么要 exp?因为 logits 可能有正有负、数量级不一,
     * exp 把每个都变成正数,再归一化,就得到"相对大小"即概率。
     *
     * 下面的 max 处理是**数值稳定技巧**:
     * 如果某个 logits 很大(比如 1000),e^1000 会溢出成无穷大(inf)。
     * 但 softmax 对整体平移不敏感:把每个 logits 都减去最大值,结果不变,
     * 却能把指数控制在小范围内,避免溢出。这行 max 就是去找那个最大值。 */
    float max = logits[0];
    for (int k = 1; k < OUTPUT_SIZE; ++k)
        if (logits[k] > max) max = logits[k];

    /* 先对每个 logits 减 max 再取 exp,累加得分母 sum */
    float sum = 0.0f;
    for (int k = 0; k < OUTPUT_SIZE; ++k) { probs[k] = expf(logits[k] - max); sum += probs[k]; }
    /* 每个 exp 结果除以 sum,得到概率,保证 probs 和为 1 */
    for (int k = 0; k < OUTPUT_SIZE; ++k)  probs[k] /= sum;
}

/* ------------- 单个样本: 前向 + 反向传播,把梯度加进 g ------------- */

/* 这是最难的一段:反向传播(backpropagation)。
 *
 * 目标:对一个样本,算出"每个权重该往哪改,才能让这次预测更准"。
 *
 * 核心思想是**链式法则**。用大白话讲:
 *   "最终错的多少"取决于"输出层错多少",
 *   "输出层错多少"取决于"隐藏层2的输出"和"输出层权重",
 *   "隐藏层2的输出"又取决于"隐藏层1的输出"和"隐藏层2权重"...
 *
 * 所以从最后往前,一层层把"误差"往回传。这就是"反向传播"名字的由来。
 *
 * delta3 / delta2 / delta1 就是每一层的"误差信号"(该层每个单元错多少)。
 * 有了它,就能算梯度 = 误差信号 x 前一层输出。
 * */
static void accumulate_one(Network *net, Gradients *g,
                           const float *input, int label)
{
    /* 全部用栈上局部数组,保证 OpenMP 并行时各线程互不干扰。
     * (如果误用 static,多个线程会同时读写同一块内存,数据就乱了) */
    float h1[HIDDEN1_SIZE], h2[HIDDEN2_SIZE];
    float delta3[OUTPUT_SIZE], delta2[HIDDEN2_SIZE], delta1[HIDDEN1_SIZE];

    /* ======= 第 1 步:前向传播,缓存每一层的输出(后面反向要用) ======= */
    for (int j = 0; j < HIDDEN1_SIZE; ++j)
        h1[j] = fmaxf(0.0f, net->b1[j] + dot(INPUT_SIZE, &net->w1[j * INPUT_SIZE], input));
    for (int j = 0; j < HIDDEN2_SIZE; ++j)
        h2[j] = fmaxf(0.0f, net->b2[j] + dot(HIDDEN1_SIZE, &net->w2[j * HIDDEN1_SIZE], h1));

    /* 输出层的 logits 和概率 */
    float logits[OUTPUT_SIZE];
    for (int k = 0; k < OUTPUT_SIZE; ++k)
        logits[k] = net->b3[k] + dot(HIDDEN2_SIZE, &net->w3[k * HIDDEN2_SIZE], h2);

    /* softmax(和 network_forward 里一样) */
    float max = logits[0];
    for (int k = 1; k < OUTPUT_SIZE; ++k)
        if (logits[k] > max) max = logits[k];
    float probs[OUTPUT_SIZE], sum = 0.0f;
    for (int k = 0; k < OUTPUT_SIZE; ++k) { probs[k] = expf(logits[k] - max); sum += probs[k]; }
    for (int k = 0; k < OUTPUT_SIZE; ++k) probs[k] /= sum;

    /* ======= 第 2 步:算输出层的误差 delta3 ======= */
    /* 这里用了一个经典结论:
     * 输出层用的是 softmax,损失用的是交叉熵,两者组合起来,
     * 导数恰好就是 "probs 减去 one-hot"(one-hot 指正确类别位置是1,其余是0)。
     *
     * 为什么这么漂亮?因为 softmax+交叉熵 求导后,复杂项互相抵消,
     * 只剩 probs[k] - (k==label)。这就是传说中的"梯度很干净"。
     *
     * 举例:正确类别是 1(圆形),预测 probs = [0.2, 0.8]。
     *        delta3[0] = 0.2 - 0 = 0.2  (该类别不是真实的,预测多了要往下调)
     *        delta3[1] = 0.8 - 1 = -0.2 (该类别是真的,预测不够要往上调)
     * 那个 (k == label ? 1.0f : 0.0f) 就是 one-hot 编码。 */
    for (int k = 0; k < OUTPUT_SIZE; ++k)
        delta3[k] = probs[k] - (k == label ? 1.0f : 0.0f);

    /* ======= 第 3 步:把输出层误差"传回"隐藏层2,得 delta2 ======= */
    /* 反向传播的关键:本层误差 = 上一层误差 经权重"投影"回来,再乘本层激活函数的导数。
     *
     * 这里求的是 delta2:
     *   gsum = Σ_k (w3[k][j] * delta3[k])  把输出层误差按权重分摊回隐藏单元 j
     *   delta2 = gsum * ReLU导数(h2[j])
     *
     * ReLU(x)=max(0,x) 的导数:x>0 时是 1,x<0 时是 0。
     * 所以 (h2[j] > 0.0f) ? gsum : 0.0f 就是乘 ReLU 导数。
     * 含义:如果这个隐藏单元被 ReLU 激活了(输出>0),它就要为误差负责;
     *       如果被 ReLU 截成 0 了(输出<=0),它这个方向不学,梯度置 0。 */
    for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        float gsum = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; ++k)
            gsum += net->w3[k * HIDDEN2_SIZE + j] * delta3[k];
        delta2[j] = (h2[j] > 0.0f) ? gsum : 0.0f;
    }

    /* ======= 第 4 步:同理,把 delta2 传回隐藏层1,得 delta1 ======= */
    for (int j = 0; j < HIDDEN1_SIZE; ++j) {
        float gsum = 0.0f;
        for (int k = 0; k < HIDDEN2_SIZE; ++k)
            gsum += net->w2[k * HIDDEN1_SIZE + j] * delta2[k];
        delta1[j] = (h1[j] > 0.0f) ? gsum : 0.0f;
    }

    /* ======= 第 5 步:用"误差信号 x 前层输出"累积梯度 =======
     * 梯度公式: dLoss/dW = 本层误差 delta x 上一层的输出。
     *
     * 也就是:要让这个权重改多少,取决于"它连的那个神经元错多少(delta)"和
     * "喂给它的输入有多大(上一层输出)"。这很直觉:
     * 一个权重的重要程度 = 误差 x 输入强度。
     *
     * 这里统一加进 g,而不是直接改权重(改权重是 network_update 的事,
     * 要等整批累加完再一起改,更稳)。 */
    for (int k = 0; k < OUTPUT_SIZE; ++k) {
        g->gb3[k] += delta3[k];
        for (int j = 0; j < HIDDEN2_SIZE; ++j)
            g->gw3[k * HIDDEN2_SIZE + j] += delta3[k] * h2[j];       /* 输出层:delta x h2 */
    }
    for (int k = 0; k < HIDDEN2_SIZE; ++k) {
        g->gb2[k] += delta2[k];
        for (int j = 0; j < HIDDEN1_SIZE; ++j)
            g->gw2[k * HIDDEN1_SIZE + j] += delta2[k] * h1[j];       /* 隐藏层2:delta2 x h1 */
    }
    for (int k = 0; k < HIDDEN1_SIZE; ++k) {
        g->gb1[k] += delta1[k];
        for (int j = 0; j < INPUT_SIZE; ++j)
            g->gw1[k * INPUT_SIZE + j] += delta1[k] * input[j];      /* 隐藏层1:delta1 x input */
    }
}

/* ------------- 批量并行计算梯度 ------------- */

/* 对一批样本(共 count 个)并行算梯度,全部累加进 g,并返回该批平均损失。
 *
 * ======= 为什么要"一批一批"而不是"一个个"学? =======
 * 如果一个样本就更新一次权重,叫"在线学习",虽然快但每一步都只基于一个样本,
 * 容易来回震荡、不稳定。
 * 如果一批(比如 64 个)的梯度累加后再更新一次,叫"批处理",更稳定,
 * 梯度是这批样本的平均方向,更有代表性。
 *
 * ======= 多线程怎么并行? =======
 * #pragma omp parallel for 是 OpenMP 的指令,意思是:
 *   把这个 for 循环的每个迭代分给不同的 CPU 线程去算。
 * 本机有 8 核,就把 count 个样本分给最多 8 个线程同时算,
 * 每个线程处理不同的样本,互不干扰,速度接近 8 倍。
 *
 * schedule(static): 把迭代平均分块给线程(静态分配),适合每个迭代耗时差不多的情况。
 * reduction(+:loss_sum): 每个线程各自累加一个局部 loss_sum,最后自动合并相加。
 *          因为 loss_sum 是"求和",多个线程各算一部分再相加,结果正确。
 *
 * ======= 为什么这样并行是安全的?(关键) =======
 * 如果多个线程同时往共享的 g->gw1[i] 做 +=,会产生"数据竞争":
 *   两个线程读同一个旧值、各自加、再写回,后写的会覆盖先写的,结果随机、时对时错。
 * 这是未定义行为,不能依赖它"碰巧能跑"。
 *
 * 正确做法(下面就是):
 *   每个线程在并行区里声明一份**自己私有**的梯度缓冲(gw1_t...),
 *   只累加自己分到的样本,绝不碰别人的,所以完全没有竞争。
 *   等所有线程算完,再用 #pragma omp critical 保证同一时刻只有一个线程
 *   把各自的私有缓冲累加进共享的 g。
 * 这样既享受多线程速度,又保证了正确性。 */
float network_accumulate_batch(Network *net, Gradients *g,
                               const float *input, const int *labels, int count)
{
    /* 先用 memset 把整个梯度缓冲清零,准备从 0 开始累加。
     * memset 把一段内存的所有字节都设成同一个值(这里是 0)。
     * sizeof(float)*数量 是这段内存的字节数。 */
    memset(g->gw1, 0, sizeof(float) * HIDDEN1_SIZE * INPUT_SIZE);
    memset(g->gb1, 0, sizeof(float) * HIDDEN1_SIZE);
    memset(g->gw2, 0, sizeof(float) * HIDDEN2_SIZE * HIDDEN1_SIZE);
    memset(g->gb2, 0, sizeof(float) * HIDDEN2_SIZE);
    memset(g->gw3, 0, sizeof(float) * OUTPUT_SIZE * HIDDEN2_SIZE);
    memset(g->gb3, 0, sizeof(float) * OUTPUT_SIZE);

    float loss_sum = 0.0f;

    /* 并行区:下面的代码会被多个线程同时执行。 */
    #pragma omp parallel reduction(+:loss_sum)
    {
        /* 每个线程分配一份**私有**梯度缓冲(大小和共享的一样),
         * 各自只累加自己那份,绝不碰别人的,也就没有数据竞争。
         * 这些数组在并行区里声明,天然是各线程私有的栈变量。 */
        float gw1_t[HIDDEN1_SIZE * INPUT_SIZE];
        float gb1_t[HIDDEN1_SIZE];
        float gw2_t[HIDDEN2_SIZE * HIDDEN1_SIZE];
        float gb2_t[HIDDEN2_SIZE];
        float gw3_t[OUTPUT_SIZE * HIDDEN2_SIZE];
        float gb3_t[OUTPUT_SIZE];
        memset(gw1_t, 0, sizeof(gw1_t));
        memset(gb1_t, 0, sizeof(gb1_t));
        memset(gw2_t, 0, sizeof(gw2_t));
        memset(gb2_t, 0, sizeof(gb2_t));
        memset(gw3_t, 0, sizeof(gw3_t));
        memset(gb3_t, 0, sizeof(gb3_t));

        /* omp for:把 [0, count) 这个循环平均分配给当前并行区的各线程,
         * 每个线程只处理分到的那部分样本。schedule(static) 是均匀静态分配。 */
        #pragma omp for schedule(static)
        for (int s = 0; s < count; ++s) {
            /* 把本线程的私有梯度传给 accumulate_one,
             * 所有梯度都加在私有缓冲上,不碰共享的 g。 */
            Gradients local = { gw1_t, gb1_t, gw2_t, gb2_t, gw3_t, gb3_t };
            accumulate_one(net, &local, &input[s * INPUT_SIZE], labels[s]);

            /* 单独算一次这个样本的损失(交叉熵),用于监控训练进度。
             * 损失 = -ln(模型给正确类别的概率)。
             * 概率越接近 1,损失越接近 0;概率越小(越错),损失越大。
             * 加 1e-9 防止 log(0) 出现负无穷。 */
            float probs[OUTPUT_SIZE];
            network_forward(net, &input[s * INPUT_SIZE], probs);
            loss_sum += -logf(probs[labels[s]] + 1e-9f);
        }

        /* 各线程算完自己的私有梯度后,合并进共享的 g。
         * #pragma omp critical 保证同一时刻只有一个线程进入合并,
         * 避免多个线程同时写 g 造成竞争。 */
        #pragma omp critical
        {
            for (int i = 0; i < HIDDEN1_SIZE * INPUT_SIZE; ++i) g->gw1[i] += gw1_t[i];
            for (int i = 0; i < HIDDEN1_SIZE; ++i)            g->gb1[i] += gb1_t[i];
            for (int i = 0; i < HIDDEN2_SIZE * HIDDEN1_SIZE; ++i) g->gw2[i] += gw2_t[i];
            for (int i = 0; i < HIDDEN2_SIZE; ++i)            g->gb2[i] += gb2_t[i];
            for (int i = 0; i < OUTPUT_SIZE * HIDDEN2_SIZE; ++i) g->gw3[i] += gw3_t[i];
            for (int i = 0; i < OUTPUT_SIZE; ++i)             g->gb3[i] += gb3_t[i];
        }
    }

    /* 返回平均损失 */
    return loss_sum / (float)count;
}

/* ------------- 用梯度更新权重(带动量) ------------- */

/* 前面累加好了梯度,现在用它来更新权重。
 *
 * ======= 普通梯度下降 =======
 *   新权重 = 旧权重 - 学习率 x 梯度
 * 意思是:梯度指向"损失上升"的方向,所以我们朝反方向(-)走一小步(学习率)。
 * 学习率 lr 就是"每一步走多大"。太大容易矫枉过正,太小学得慢。
 *
 * ======= 什么是"动量"(momentum)? =======
 * 想象一个小球滚下山坡。普通梯度下降每一步只看当前坡度,
 * 就像每一步都刹住重新判断,遇到小坑容易卡住。
 * 动量 = 给小球一个"惯性":它记住之前滚的方向,新方向 = 之前的方向 x 动量系数 + 当前梯度。
 *
 *   速度 v = momentum * 上次速度 - lr * 梯度
 *   权重 += v
 *
 * momentum=0.9 表示保留 90% 的旧方向,只加当前梯度的 10%。
 * 好处:
 *   - 在平坦处能靠惯性冲过去,不那么容易陷入局部极小值。
 *   - 在方向一致时加速,学得更快。
 *   - 在方向上经常变化时能抑制震荡。
 *
 * 这里乘以 scale(=1/count) 是为了取"平均梯度"(这批有 count 个样本,
 * 前面累加的是总和,除以 count 得到平均)。
 * */
void network_update(Network *net, Gradients *g, float lr, float momentum, int count)
{
    const float scale = 1.0f / (float)count;   /* 平均梯度:把累加和除以样本数 */

    /* 下面 6 个并行 for,分别更新 3 层各自的权重和偏置。
     * 每个数组元素独立更新,互不依赖,所以可以并行(用 OpenMP)。
     *
     * 以 w1 为例:
     *   net->vw1[i] = momentum*vw1[i] - lr*(gw1[i]*scale)  更新速度(带惯性)
     *   net->w1[i]  += vw1[i]                               用速度去更新权重
     * 注意 vb/vw 是"速度缓冲",存的是"上一步改了多少方向",
     * 它自己也会被更新,相当于记住了历史。 */

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < HIDDEN1_SIZE * INPUT_SIZE; ++i) {
        net->vw1[i] = momentum * net->vw1[i] - lr * (g->gw1[i] * scale);
        net->w1[i]  += net->vw1[i];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < HIDDEN1_SIZE; ++i) {
        net->vb1[i] = momentum * net->vb1[i] - lr * (g->gb1[i] * scale);
        net->b1[i]  += net->vb1[i];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < HIDDEN2_SIZE * HIDDEN1_SIZE; ++i) {
        net->vw2[i] = momentum * net->vw2[i] - lr * (g->gw2[i] * scale);
        net->w2[i]  += net->vw2[i];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < HIDDEN2_SIZE; ++i) {
        net->vb2[i] = momentum * net->vb2[i] - lr * (g->gb2[i] * scale);
        net->b2[i]  += net->vb2[i];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN2_SIZE; ++i) {
        net->vw3[i] = momentum * net->vw3[i] - lr * (g->gw3[i] * scale);
        net->w3[i]  += net->vw3[i];
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        net->vb3[i] = momentum * net->vb3[i] - lr * (g->gb3[i] * scale);
        net->b3[i]  += net->vb3[i];
    }
}

/* 预测:对一张输入图,返回判别结果。
 * 比较两个类别的概率,概率大的那个就是预测结果。
 * 返回 1 = 圆形,返回 0 = 矩形。 */
int network_predict(Network *net, const float *input)
{
    float probs[OUTPUT_SIZE];
    network_forward(net, input, probs);
    return probs[1] > probs[0] ? 1 : 0;
}

/* ------------- 保存/加载模型(训练成果的持久化) ------------- */

/* 保存:把权重写进二进制文件。
 * 为什么要存?因为训练要花时间。训练一次得到好权重后存到硬盘,
 * 下次直接读回来就能用(predict 模式),不用重新训练。
 *
 * 文件格式:先写 6 个 int 记录各层维度,再按顺序写所有权重/偏置。
 * 读的时候按同样的顺序读回来,就能完整还原网络。
 * fwrite(ptr, size, count, f):把 ptr 指向的 count 个 size 字节写进文件 f。 */
int network_save(const Network *net, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");   /* "wb" = write 二进制(可含任意字节,不会乱改) */
    if (!f) return -1;

    int dims[6] = { INPUT_SIZE, HIDDEN1_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE };
    fwrite(dims, sizeof(int), 6, f);                       /* 先写维度,读的时候校验用 */
    fwrite(net->w1, sizeof(float), HIDDEN1_SIZE * INPUT_SIZE, f);
    fwrite(net->b1, sizeof(float), HIDDEN1_SIZE, f);
    fwrite(net->w2, sizeof(float), HIDDEN2_SIZE * HIDDEN1_SIZE, f);
    fwrite(net->b2, sizeof(float), HIDDEN2_SIZE, f);
    fwrite(net->w3, sizeof(float), OUTPUT_SIZE * HIDDEN2_SIZE, f);
    fwrite(net->b3, sizeof(float), OUTPUT_SIZE, f);

    fclose(f);
    return 0;
}

/* 加载:从二进制文件读回权重,还原一个网络。
 * 和 save 完全对称。返回 NULL 表示读取失败(比如文件不存在)。
 * */
Network *network_load(const char *file_path)
{
    FILE *f = fopen(file_path, "rb");   /* "rb" = read 二进制 */
    if (!f) return NULL;

    int dims[6];
    /* 先读 6 个维度。fread 返回实际读到的元素个数,如果读不到 6 个说明文件坏了 */
    if (fread(dims, sizeof(int), 6, f) != 6) { fclose(f); return NULL; }

    Network *net = network_create();    /* 先创建(会分配内存并随机初始化) */
    /* 再把文件里的权重覆盖上去(读回来的值才是有用的训练成果) */
    fread(net->w1, sizeof(float), HIDDEN1_SIZE * INPUT_SIZE, f);
    fread(net->b1, sizeof(float), HIDDEN1_SIZE, f);
    fread(net->w2, sizeof(float), HIDDEN2_SIZE * HIDDEN1_SIZE, f);
    fread(net->b2, sizeof(float), HIDDEN2_SIZE, f);
    fread(net->w3, sizeof(float), OUTPUT_SIZE * HIDDEN2_SIZE, f);
    fread(net->b3, sizeof(float), OUTPUT_SIZE, f);

    fclose(f);
    return net;
}

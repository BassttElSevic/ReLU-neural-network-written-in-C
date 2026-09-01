#include "network.h"
#include "rng.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/* ------------- 工具 ------------- */

static float *alloc(int n) { return (float *)calloc((size_t)n, sizeof(float)); }

static float dot(int n, const float *a, const float *b)
{
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

/* 用均匀分布初始化权重,范围 [-gain, gain],gain 取 1/sqrt(fan_in)。 */
static void init_weights(float *w, int fan_in, int fan_out)
{
    float gain = 1.0f / sqrtf((float)fan_in);
    for (int i = 0; i < fan_in * fan_out; ++i)
        w[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * gain;
}

/* ------------- 生命周期 ------------- */

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

    net->vw1 = alloc(HIDDEN1_SIZE * INPUT_SIZE);
    net->vb1 = alloc(HIDDEN1_SIZE);
    net->vw2 = alloc(HIDDEN2_SIZE * HIDDEN1_SIZE);
    net->vb2 = alloc(HIDDEN2_SIZE);
    net->vw3 = alloc(OUTPUT_SIZE * HIDDEN2_SIZE);
    net->vb3 = alloc(OUTPUT_SIZE);

    init_weights(net->w1, INPUT_SIZE, HIDDEN1_SIZE);
    init_weights(net->w2, HIDDEN1_SIZE, HIDDEN2_SIZE);
    init_weights(net->w3, HIDDEN2_SIZE, OUTPUT_SIZE);

    return net;
}

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

/* ------------- 前向 ------------- */

void network_forward(Network *net, const float *input, float *probs)
{
    float h1[HIDDEN1_SIZE], h2[HIDDEN2_SIZE], logits[OUTPUT_SIZE];

    for (int j = 0; j < HIDDEN1_SIZE; ++j)
        h1[j] = fmaxf(0.0f, net->b1[j] + dot(INPUT_SIZE, &net->w1[j * INPUT_SIZE], input));

    for (int j = 0; j < HIDDEN2_SIZE; ++j)
        h2[j] = fmaxf(0.0f, net->b2[j] + dot(HIDDEN1_SIZE, &net->w2[j * HIDDEN1_SIZE], h1));

    for (int k = 0; k < OUTPUT_SIZE; ++k)
        logits[k] = net->b3[k] + dot(HIDDEN2_SIZE, &net->w3[k * HIDDEN2_SIZE], h2);

    float max = logits[0];
    for (int k = 1; k < OUTPUT_SIZE; ++k)
        if (logits[k] > max) max = logits[k];

    float sum = 0.0f;
    for (int k = 0; k < OUTPUT_SIZE; ++k) { probs[k] = expf(logits[k] - max); sum += probs[k]; }
    for (int k = 0; k < OUTPUT_SIZE; ++k)  probs[k] /= sum;
}

/* ------------- 单个样本: 前向 + 反向,把梯度加入 g ------------- */

static void accumulate_one(Network *net, Gradients *g,
                           const float *input, int label)
{
    /* 全部用栈上局部数组,保证 OpenMP 并行时各线程互不干扰 */
    float h1[HIDDEN1_SIZE], h2[HIDDEN2_SIZE];
    float delta3[OUTPUT_SIZE], delta2[HIDDEN2_SIZE], delta1[HIDDEN1_SIZE];

    for (int j = 0; j < HIDDEN1_SIZE; ++j)
        h1[j] = fmaxf(0.0f, net->b1[j] + dot(INPUT_SIZE, &net->w1[j * INPUT_SIZE], input));
    for (int j = 0; j < HIDDEN2_SIZE; ++j)
        h2[j] = fmaxf(0.0f, net->b2[j] + dot(HIDDEN1_SIZE, &net->w2[j * HIDDEN1_SIZE], h1));

    float logits[OUTPUT_SIZE];
    for (int k = 0; k < OUTPUT_SIZE; ++k)
        logits[k] = net->b3[k] + dot(HIDDEN2_SIZE, &net->w3[k * HIDDEN2_SIZE], h2);

    float max = logits[0];
    for (int k = 1; k < OUTPUT_SIZE; ++k)
        if (logits[k] > max) max = logits[k];
    float probs[OUTPUT_SIZE], sum = 0.0f;
    for (int k = 0; k < OUTPUT_SIZE; ++k) { probs[k] = expf(logits[k] - max); sum += probs[k]; }
    for (int k = 0; k < OUTPUT_SIZE; ++k) probs[k] /= sum;

    /* 输出层误差: softmax + 交叉熵 梯度 = probs - one_hot */
    for (int k = 0; k < OUTPUT_SIZE; ++k)
        delta3[k] = probs[k] - (k == label ? 1.0f : 0.0f);

    for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        float gsum = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; ++k)
            gsum += net->w3[k * HIDDEN2_SIZE + j] * delta3[k];
        delta2[j] = (h2[j] > 0.0f) ? gsum : 0.0f;
    }

    for (int j = 0; j < HIDDEN1_SIZE; ++j) {
        float gsum = 0.0f;
        for (int k = 0; k < HIDDEN2_SIZE; ++k)
            gsum += net->w2[k * HIDDEN1_SIZE + j] * delta2[k];
        delta1[j] = (h1[j] > 0.0f) ? gsum : 0.0f;
    }

    /* 累积梯度 */
    for (int k = 0; k < OUTPUT_SIZE; ++k) {
        g->gb3[k] += delta3[k];
        for (int j = 0; j < HIDDEN2_SIZE; ++j)
            g->gw3[k * HIDDEN2_SIZE + j] += delta3[k] * h2[j];
    }
    for (int k = 0; k < HIDDEN2_SIZE; ++k) {
        g->gb2[k] += delta2[k];
        for (int j = 0; j < HIDDEN1_SIZE; ++j)
            g->gw2[k * HIDDEN1_SIZE + j] += delta2[k] * h1[j];
    }
    for (int k = 0; k < HIDDEN1_SIZE; ++k) {
        g->gb1[k] += delta1[k];
        for (int j = 0; j < INPUT_SIZE; ++j)
            g->gw1[k * INPUT_SIZE + j] += delta1[k] * input[j];
    }
}

/* 批量并行梯度累积。OpenMP 把批内样本分给各线程,互不干扰地累加。 */
float network_accumulate_batch(Network *net, Gradients *g,
                               const float *input, const int *labels, int count)
{
    memset(g->gw1, 0, sizeof(float) * HIDDEN1_SIZE * INPUT_SIZE);
    memset(g->gb1, 0, sizeof(float) * HIDDEN1_SIZE);
    memset(g->gw2, 0, sizeof(float) * HIDDEN2_SIZE * HIDDEN1_SIZE);
    memset(g->gb2, 0, sizeof(float) * HIDDEN2_SIZE);
    memset(g->gw3, 0, sizeof(float) * OUTPUT_SIZE * HIDDEN2_SIZE);
    memset(g->gb3, 0, sizeof(float) * OUTPUT_SIZE);

    float loss_sum = 0.0f;

    #pragma omp parallel for schedule(static) reduction(+:loss_sum)
    for (int s = 0; s < count; ++s) {
        accumulate_one(net, g, &input[s * INPUT_SIZE], labels[s]);

        /* 单独算一次这个样本的损失(前向) */
        float probs[OUTPUT_SIZE];
        network_forward(net, &input[s * INPUT_SIZE], probs);
        loss_sum += -logf(probs[labels[s]] + 1e-9f);
    }

    return loss_sum / (float)count;
}

/* 动量更新: v = momentum*v - lr*(平均梯度); 参数 += v */
void network_update(Network *net, Gradients *g, float lr, float momentum, int count)
{
    const float scale = 1.0f / (float)count;   /* 平均梯度 */

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

int network_predict(Network *net, const float *input)
{
    float probs[OUTPUT_SIZE];
    network_forward(net, input, probs);
    return probs[1] > probs[0] ? 1 : 0;
}

/* ------------- 保存/加载 ------------- */

int network_save(const Network *net, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (!f) return -1;

    int dims[6] = { INPUT_SIZE, HIDDEN1_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE };
    fwrite(dims, sizeof(int), 6, f);
    fwrite(net->w1, sizeof(float), HIDDEN1_SIZE * INPUT_SIZE, f);
    fwrite(net->b1, sizeof(float), HIDDEN1_SIZE, f);
    fwrite(net->w2, sizeof(float), HIDDEN2_SIZE * HIDDEN1_SIZE, f);
    fwrite(net->b2, sizeof(float), HIDDEN2_SIZE, f);
    fwrite(net->w3, sizeof(float), OUTPUT_SIZE * HIDDEN2_SIZE, f);
    fwrite(net->b3, sizeof(float), OUTPUT_SIZE, f);

    fclose(f);
    return 0;
}

Network *network_load(const char *file_path)
{
    FILE *f = fopen(file_path, "rb");
    if (!f) return NULL;

    int dims[6];
    if (fread(dims, sizeof(int), 6, f) != 6) { fclose(f); return NULL; }

    Network *net = network_create();
    fread(net->w1, sizeof(float), HIDDEN1_SIZE * INPUT_SIZE, f);
    fread(net->b1, sizeof(float), HIDDEN1_SIZE, f);
    fread(net->w2, sizeof(float), HIDDEN2_SIZE * HIDDEN1_SIZE, f);
    fread(net->b2, sizeof(float), HIDDEN2_SIZE, f);
    fread(net->w3, sizeof(float), OUTPUT_SIZE * HIDDEN2_SIZE, f);
    fread(net->b3, sizeof(float), OUTPUT_SIZE, f);

    fclose(f);
    return net;
}

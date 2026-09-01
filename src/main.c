#include "layer.h"
#include "network.h"
#include "dataset.h"
#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* 训练超参数
 * 说明: 输入维度 2500(50x50),纯 C 批量梯度下降,OpenMP 多线程。 */
#define TRAIN_SAMPLES  3000
#define VAL_SAMPLES    500
#define EPOCHS         40
#define BATCH_SIZE     64
#define LEARNING_RATE  0.02f
#define MOMENTUM       0.9f

static float train_loss_history[EPOCHS];
static float train_acc_history[EPOCHS];
static float val_acc_history[EPOCHS];

static void write_curve_csv(const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "epoch,train_loss,train_acc,val_acc\n");
    for (int e = 0; e < EPOCHS; ++e)
        fprintf(f, "%d,%.6f,%.4f,%.4f\n",
                e + 1, train_loss_history[e], train_acc_history[e], val_acc_history[e]);
    fclose(f);
    printf("[INFO] 训练曲线已写入 %s\n", path);
}

/* --------- 训练模式: 训练并保存模型 --------- */
static int cmd_train(int seed)
{
    rng_seed(seed);

    printf("=== 矩形/圆形 分类器训练 ===\n");
    printf("配置: 输入 %d 像素, 隐藏层 %d/%d, 输出 %d 类\n",
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
    printf("训练 %d 样本, 验证 %d 样本, %d 轮, 批大小 %d, lr=%.3f, 动量=%.2f, 种子=%d\n\n",
           TRAIN_SAMPLES, VAL_SAMPLES, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, seed);
    printf("CPU 线程数: %d (OpenMP)\n\n", omp_get_max_threads());

    Network *net = network_create();
    Gradients *grad = gradients_create();
    if (!net || !grad) { fprintf(stderr, "ERROR: 无法创建网络/梯度\n"); return 1; }

    float *batch_input = (float *)malloc(sizeof(float) * BATCH_SIZE * INPUT_SIZE);
    int   *batch_label = (int *)malloc(sizeof(int) * BATCH_SIZE);
    float *eval_input = (float *)malloc(sizeof(float) * TRAIN_SAMPLES * INPUT_SIZE);
    Layer layer;
    if (!batch_input || !batch_label || !eval_input) {
        fprintf(stderr, "ERROR: 内存分配失败\n");
        return 1;
    }

    for (int e = 0; e < EPOCHS; ++e) {
        float loss_sum = 0.0f;
        int train_correct = 0;
        int n_batches = 0;
        int count = 0;

        for (int b = 0; b < TRAIN_SAMPLES; b += BATCH_SIZE) {
            count = BATCH_SIZE;
            if (b + count > TRAIN_SAMPLES) count = TRAIN_SAMPLES - b;

            for (int s = 0; s < count; ++s) {
                int label;
                dataset_random_sample(layer, &batch_input[s * INPUT_SIZE], &label);
                batch_label[s] = label;
            }

            float loss = network_accumulate_batch(net, grad, batch_input, batch_label, count);
            network_update(net, grad, LEARNING_RATE, MOMENTUM, count);

            loss_sum += loss;
            n_batches++;
        }

        for (int s = 0; s < TRAIN_SAMPLES; ++s) {
            int label;
            dataset_random_sample(layer, &eval_input[s * INPUT_SIZE], &label);
            if (network_predict(net, &eval_input[s * INPUT_SIZE]) == label) train_correct++;
        }

        train_loss_history[e] = loss_sum / (float)n_batches;
        train_acc_history[e] = (float)train_correct / (float)TRAIN_SAMPLES;

        int val_correct = 0;
        for (int s = 0; s < VAL_SAMPLES; ++s) {
            int label;
            dataset_random_sample(layer, &eval_input[s * INPUT_SIZE], &label);
            if (network_predict(net, &eval_input[s * INPUT_SIZE]) == label) val_correct++;
        }
        val_acc_history[e] = (float)val_correct / (float)VAL_SAMPLES;

        printf("轮 %3d  |  损失 %.4f  |  训练准确率 %.3f  |  验证准确率 %.3f\n",
               e + 1, train_loss_history[e], train_acc_history[e], val_acc_history[e]);

        if (e == EPOCHS - 1 || (e + 1) % 10 == 0) {
            char path[64];
            int label;
            snprintf(path, sizeof(path), "data/sample_epoch_%02d.ppm", e + 1);
            dataset_random_sample(layer, eval_input, &label);
            layer_save_as_ppm(layer, path);
        }
    }

    if (network_save(net, "data/shape_net.bin") != 0) {
        fprintf(stderr, "ERROR: 保存模型失败\n");
    } else {
        printf("[INFO] 模型已保存到 data/shape_net.bin\n");
    }

    write_curve_csv("data/training_curve.csv");

    free(batch_input);
    free(batch_label);
    free(eval_input);
    gradients_free(grad);
    network_free(net);
    return 0;
}

/* --------- 预测模式: 加载已训练模型,不重新训练 --------- */
static int cmd_predict(void)
{
    Network *net = network_load("data/shape_net.bin");
    if (!net) {
        fprintf(stderr, "ERROR: 未找到模型 data/shape_net.bin,请先运行 shape_classifier train\n");
        return 1;
    }

    printf("=== 加载模型: data/shape_net.bin ===\n");
    printf("直接预测,不重新训练。\n\n");

    Layer layer;
    float input[INPUT_SIZE];
    int correct = 0, total = 200;

    /* 在随机样本上评估模型准确率(不更新权重) */
    for (int i = 0; i < total; ++i) {
        int label;
        dataset_random_sample(layer, input, &label);
        if (network_predict(net, input) == label) correct++;
    }
    printf("评估 %d 个随机样本, 准确率 = %.2f%%\n\n",
           total, 100.0f * (float)correct / (float)total);

    printf("=== 实际预测演示 ===\n");
    for (int i = 0; i < 6; ++i) {
        int label;
        dataset_random_sample(layer, input, &label);
        int pred = network_predict(net, input);
        const char *truth = label == 0 ? "矩形" : "圆形";
        const char *guess = pred == 0 ? "矩形" : "圆形";
        printf("样本 %d: 实际=%s, 预测=%s  %s\n", i + 1, truth, guess,
               (label == pred) ? "正确" : "错误");
    }

    network_free(net);
    return 0;
}

int main(int argc, char **argv)
{
    /* 用法:
     *   shape_classifier train [seed]  -> 训练并保存模型
     *   shape_classifier predict       -> 加载模型直接预测
     */
    const char *mode = (argc > 1) ? argv[1] : "train";

    if (strcmp(mode, "train") == 0) {
        int seed = (argc > 2) ? atoi(argv[2]) : 12345;
        return cmd_train(seed);
    } else if (strcmp(mode, "predict") == 0) {
        return cmd_predict();
    } else {
        fprintf(stderr, "用法: %s train [seed]  |  %s predict\n", argv[0], argv[0]);
        return 1;
    }
}

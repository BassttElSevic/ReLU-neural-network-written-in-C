#include "layer.h"
#include "network.h"
#include "dataset.h"
#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* 跨平台目录创建所需的头文件 */
#ifdef _WIN32
#include <direct.h>   /* Windows: _mkdir */
#else
#include <sys/stat.h> /* POSIX(macOS/Linux): mkdir */
#endif

/* ================= 训练超参数 =================
 * 这些是"调参":改它们会影响训练速度和质量。
 *   TRAIN_SAMPLES 每轮用多少样本训练(不是越多越好,够用即可)
 *   VAL_SAMPLES   验证集大小(用来测模型在新数据上准不准,防过拟合)
 *   EPOCHS        训练多少轮(把全部训练样本过一遍算一轮)
 *   BATCH_SIZE    每批一起算梯度的样本数(多线程并行也是按批分)
 *   LEARNING_RATE 学习率,每步改权重的幅度
 *   MOMENTUM      动量系数,让学习带"惯性",防局部极小值
 */
#define TRAIN_SAMPLES  3000
#define VAL_SAMPLES    500
#define EPOCHS         40
#define BATCH_SIZE     64
#define LEARNING_RATE  0.02f
#define MOMENTUM       0.9f

/* 记录每一轮的训练损失/准确率/验证准确率,训练结束后写成 CSV 供绘图。
 * static:只在当前文件可见。[] 表示数组长度由 EPOCHS 决定。 */
static float train_loss_history[EPOCHS];
static float train_acc_history[EPOCHS];
static float val_acc_history[EPOCHS];

/* 把三组历史数据写成 CSV 文件(纯文本,逗号分隔)。
 * CSV 方便被 python/matplotlib 读取来做训练曲线的可视化。
 * 文件每行:轮数, 训练损失, 训练准确率, 验证准确率 */
static void write_curve_csv(const char *path)
{
    FILE *f = fopen(path, "w");   /* "w" = write 文本(这里是纯文本记录) */
    if (!f) return;
    fprintf(f, "epoch,train_loss,train_acc,val_acc\n");   /* 表头 */
    for (int e = 0; e < EPOCHS; ++e)
        fprintf(f, "%d,%.6f,%.4f,%.4f\n",
                e + 1, train_loss_history[e], train_acc_history[e], val_acc_history[e]);
    fclose(f);
    printf("[INFO] 训练曲线已写入 %s\n", path);
}

/* 确保 data/ 目录存在。
 * 因为 .gitignore 忽略了 data/(训练产物不进版本库),用户 clone 下来后
 * 本地并没有这个目录,若程序直接往里写文件,fopen 会失败、报 open file 错误。
 * 所以在程序启动时用 mkdir 自动创建它,用户不需要手动建目录。
 * 已存在时 mkdir 会返回 -1(EEXIST),这里忽略即可。 */
static void ensure_data_dir(void)
{
#ifdef _WIN32
    /* Windows 用 _mkdir(单参数,不需要权限位),direct.h 已在文件顶部包含 */
    _mkdir("data");
#else
    /* POSIX(macOS/Linux)用 mkdir:路径 + 权限位。
     * 用 <sys/stat.h> 的符号常量表示 0755(所有者 rwx,组 r-x,其他 r-x),
     * 比裸八进制更自解释、更可移植。 */
    mkdir("data", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
#endif
}

/* ============ 训练模式: 训练并保存模型 ============ */
static int cmd_train(int seed)
{
    rng_seed(seed);   /* 设随机种子,让本次训练可复现 */

    printf("=== 矩形/圆形 分类器训练 ===\n");
    printf("配置: 输入 %d 像素, 隐藏层 %d/%d, 输出 %d 类\n",
           INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE);
    printf("训练 %d 样本, 验证 %d 样本, %d 轮, 批大小 %d, lr=%.3f, 动量=%.2f, 种子=%d\n\n",
           TRAIN_SAMPLES, VAL_SAMPLES, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, seed);
    /* omp_get_max_threads(): 返回 OpenMP 当前会用多少个 CPU 线程 */
    printf("CPU 线程数: %d (OpenMP)\n\n", omp_get_max_threads());

    /* 创建网络(随机初始化权重)和梯度累加器 */
    Network *net = network_create();
    Gradients *grad = gradients_create();
    if (!net || !grad) { fprintf(stderr, "ERROR: 无法创建网络/梯度\n"); return 1; }

    /* 用 malloc 在堆上分配大块内存(训练样本可能很多,栈放不下)。
     *   batch_input: 一批样本的输入,大小 BATCH_SIZE x 2500
     *   batch_label: 一批样本的标签
     *   eval_input:  用于评估准确率的样本(数量够 TRAIN_SAMPLES 用)
     * Layer layer:  一个临时图层,用来生成样本再拉平 */
    float *batch_input = (float *)malloc(sizeof(float) * BATCH_SIZE * INPUT_SIZE);
    int   *batch_label = (int *)malloc(sizeof(int) * BATCH_SIZE);
    float *eval_input = (float *)malloc(sizeof(float) * TRAIN_SAMPLES * INPUT_SIZE);
    Layer layer;
    if (!batch_input || !batch_label || !eval_input) {
        fprintf(stderr, "ERROR: 内存分配失败\n");
        return 1;
    }

    /* ======= 主训练循环: 每一轮 ======= */
    for (int e = 0; e < EPOCHS; ++e) {
        float loss_sum = 0.0f;         /* 累计本轮所有批的损失 */
        int train_correct = 0;         /* 训练集预测正确的样本数 */
        int n_batches = 0;             /* 本轮处理了几批 */
        int count = 0;

        /* ======= 内层: 把 TRAIN_SAMPLES 个样本分成若干批 ======= */
        for (int b = 0; b < TRAIN_SAMPLES; b += BATCH_SIZE) {
            count = BATCH_SIZE;                                  /* 本批大小 */
            if (b + count > TRAIN_SAMPLES) count = TRAIN_SAMPLES - b; /* 最后一批可能不满 */

            /* 先为这一批生成样本(随机矩形/圆形)并存进 batch 数组 */
            for (int s = 0; s < count; ++s) {
                int label;
                dataset_random_sample(layer, &batch_input[s * INPUT_SIZE], &label);
                batch_label[s] = label;
            }

            /* 并行算这一批的梯度(多线程,见 network.c) */
            float loss = network_accumulate_batch(net, grad, batch_input, batch_label, count);
            /* 用梯度更新权重(带动量) */
            network_update(net, grad, LEARNING_RATE, MOMENTUM, count);

            loss_sum += loss;
            n_batches++;
        }

        /* ======= 算训练集准确率 ======= */
        for (int s = 0; s < TRAIN_SAMPLES; ++s) {
            int label;
            dataset_random_sample(layer, &eval_input[s * INPUT_SIZE], &label);
            if (network_predict(net, &eval_input[s * INPUT_SIZE]) == label) train_correct++;
        }

        train_loss_history[e] = loss_sum / (float)n_batches;             /* 平均损失 */
        train_acc_history[e] = (float)train_correct / (float)TRAIN_SAMPLES; /* 训练准确率 */

        /* ======= 算验证集准确率(用独立随机样本,只预测不更新权重) =======
         * 验证集用来衡量"模型在新数据上表现如何",防止过拟合。
         * (过拟合 = 训练集上很准,但没见过的新数据上很差) */
        int val_correct = 0;
        for (int s = 0; s < VAL_SAMPLES; ++s) {
            int label;
            dataset_random_sample(layer, &eval_input[s * INPUT_SIZE], &label);
            if (network_predict(net, &eval_input[s * INPUT_SIZE]) == label) val_correct++;
        }
        val_acc_history[e] = (float)val_correct / (float)VAL_SAMPLES;

        /* 打印本轮进度 */
        printf("轮 %3d  |  损失 %.4f  |  训练准确率 %.3f  |  验证准确率 %.3f\n",
               e + 1, train_loss_history[e], train_acc_history[e], val_acc_history[e]);

        /* 每 10 轮(和最后一轮)保存一张样例图,方便肉眼确认生成的数据 */
        if (e == EPOCHS - 1 || (e + 1) % 10 == 0) {
            char path[64];
            int label;
            snprintf(path, sizeof(path), "data/sample_epoch_%02d.ppm", e + 1);
            dataset_random_sample(layer, eval_input, &label);
            layer_save_as_ppm(layer, path);
        }
    }

    /* ======= 训练结束,保存模型 ======= */
    if (network_save(net, "data/shape_net.bin") != 0) {
        fprintf(stderr, "ERROR: 保存模型失败\n");
    } else {
        printf("[INFO] 模型已保存到 data/shape_net.bin\n");
    }

    write_curve_csv("data/training_curve.csv");   /* 保存训练曲线数据 */

    /* 释放内存(和 malloc 成对使用,防泄漏) */
    free(batch_input);
    free(batch_label);
    free(eval_input);
    gradients_free(grad);
    network_free(net);
    return 0;
}

/* ============ 预测模式: 加载已训练模型,不重新训练 ============ */
static int cmd_predict(void)
{
    /* 从硬盘读回训练好的权重。如果文件不存在则报错。
     * 这就是"模型可复用,不用每次重训"的核心。 */
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

    /* 用 200 个随机样本评估模型准确率(只预测,不改权重) */
    for (int i = 0; i < total; ++i) {
        int label;
        dataset_random_sample(layer, input, &label);
        if (network_predict(net, input) == label) correct++;
    }
    printf("评估 %d 个随机样本, 准确率 = %.2f%%\n\n",
           total, 100.0f * (float)correct / (float)total);

    /* 展示几个具体预测结果 */
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

    network_free(net);   /* 用完释放 */
    return 0;
}

/* ============ 程序入口 ============ */
int main(int argc, char **argv)
{
    /* argc = 命令行参数个数,argv = 参数数组。
     *   ./shape_classifier              -> argc=1, mode 默认 "train"
     *   ./shape_classifier train 12345  -> argc=3, mode="train", seed=12345
     *   ./shape_classifier predict      -> argc=2, mode="predict"
     * argv[0] 永远是程序自己的名字。 */
    const char *mode = (argc > 1) ? argv[1] : "train";

    /* 先确保 data/ 目录存在,再执行训练/预测。
     * 这样用户 clone 下来直接运行即可,无需手动建目录。 */
    ensure_data_dir();

    /* strcmp 比较字符串是否相等。等于 0 表示相等。 */
    if (strcmp(mode, "train") == 0) {
        int seed = (argc > 2) ? atoi(argv[2]) : 12345;   /* atoi: 字符串转整数 */
        return cmd_train(seed);
    } else if (strcmp(mode, "predict") == 0) {
        return cmd_predict();
    } else {
        fprintf(stderr, "用法: %s train [seed]  |  %s predict\n", argv[0], argv[0]);
        return 1;   /* 非 0 返回值表示程序异常退出 */
    }
}

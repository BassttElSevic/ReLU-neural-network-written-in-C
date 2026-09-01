#include "dataset.h"
#include "rng.h"
#include "network.h"
#include "layer.h"
#include <assert.h>

/* 把一张二维图层(Layer,50x50)拉平成一维数组(2500 个 float)。
 *
 * 为什么要拉平?因为神经网络的全连接层期望输入是一维向量。
 * 二维图(行 x 列)需要按固定顺序展成一行,这里按"逐行"的顺序:
 *   第 0 行的 50 个 -> 下标 0~49
 *   第 1 行的 50 个 -> 下标 50~99
 *   ...
 * 换算公式:flat[y * WIDTH + x]。
 * 这是 C 里把二维数组当一维数组访问的标准手法。
 * */
void dataset_flatten(Layer layer, float *flat)
{
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            flat[y * WIDTH + x] = layer[y][x];
}

/* 生成一个随机样本:随机画一个矩形或一个圆形。
 *
 * 这个函数是"训练数据来源"。因为程序能自己生成样本,
 * 所以不需要人工标注几千张图。它每次随机决定画矩形还是画圆形,
 * 随机位置、随机大小,然后填进 layer。
 *
 * 输出:
 *   layer  - 画好形状的图层(50x50,像素 0 或 1)
 *   flat   - 拉平后的一维数组(喂给神经网络)
 *   *label - 类别标签:0=矩形,1=圆形
 * */
void dataset_random_sample(Layer layer, float *flat, int *label)
{
    /* rand_range(0, 2) 返回 0 或 1,等概率二选一 */
    if (rand_range(0, 2) == 0) {
        /* ---- 随机矩形 ---- */
        layer_clear(layer, 0.0f);   /* 先清空成黑背景 */

        int x = rand_range(0, WIDTH);     /* 左上角横坐标 0~49 */
        int y = rand_range(0, HEIGHT);    /* 左上角纵坐标 0~49 */

        /* 宽度不能超过"从 x 到右边界还剩多少",否则矩形会画到图片外面。
         * 用 WIDTH - x 作为限制。 */
        int w = WIDTH - x;
        if (w < 2) w = 2;          /* 保底:至少宽 2,避免矩形窄成一条线 */
        w = rand_range(1, w + 1);  /* 在 [1, WIDTH-x] 里随机宽度 */

        /* 高度同理,受 y 到图片下边界的距离限制 */
        int h = HEIGHT - y;
        if (h < 2) h = 2;
        h = rand_range(1, h + 1);

        layer_fill_rectangle(layer, x, y, w, h, 1.0f);  /* 画白色矩形 */
        *label = 0;
    } else {
        /* ---- 随机圆形 ---- */
        layer_clear(layer, 0.0f);

        int cx = rand_range(0, WIDTH);    /* 圆心横坐标 */
        int cy = rand_range(0, HEIGHT);   /* 圆心纵坐标 */

        /* 半径上限:取"圆心到四个边界距离的最小值"。
         * 因为圆心可能很靠边,如果半径太大圆就会越界。
         * 用一连串 if 找出最小的那个距离,这就是安全的最大半径。 */
        int r_max = cx;                 /* 到左边界距离 */
        if (cy < r_max)            r_max = cy;         /* 到上边界距离(取更小) */
        if (WIDTH - cx < r_max)    r_max = WIDTH - cx; /* 到右边界距离 */
        if (HEIGHT - cy < r_max)   r_max = HEIGHT - cy;/* 到下边界距离 */
        if (r_max < 2)             r_max = 2;          /* 保底半径至少 2 */

        int r = rand_range(2, r_max + 1);  /* 在 [2, 安全最大半径] 随机 */
        layer_fill_circle(layer, cx, cy, r, 1.0f);  /* 画白色实心圆 */
        *label = 1;
    }

    dataset_flatten(layer, flat);   /* 最后把图层拉平成一维,供网络使用 */
}

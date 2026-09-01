#include "layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/* 把整数 x 限制在 [low, high] 区间内。
 * 用途:绘图时坐标可能超出图片边界(比如矩形圆心很靠边、半径又大),
 *      用这个函数把坐标"夹"回合法范围,防止数组越界访问。
 * 这是图形代码里很常用的工具。 */
int clamp_int(int x, int low, int high)
{
    if (x < low)   x = low;   /* 比下界小,顶到下界 */
    if (x > high)  x = high;  /* 比上界大,顶到上界 */
    return x;
}

/* 把整个图层填充成同一个值。
 * value=0.0f 相当于"清空成黑背景",value=1.0f 相当于"填满白色"。
 * 双重循环遍历每个像素。 */
void layer_clear(Layer layer, float value)
{
    for (int y = 0; y < HEIGHT; ++y)      /* y 是行,从 0 到 HEIGHT-1 */
        for (int x = 0; x < WIDTH; ++x)   /* x 是列,从 0 到 WIDTH-1 */
            layer[y][x] = value;          /* C 里二维数组用 [行][列] 访问 */
}

/* 在图层上填充一个矩形区域。
 * (x, y) 为左上角,宽 w、高 h,所有落在图片内的格子都置为 value。
 *
 * 为什么用 clamp_int 夹四个坐标?
 * 因为随机生成的矩形可能有一部分"画到图片外面"(比如左上角 x=18、宽 10,
 * 那右边缘就是 27,超出 WIDTH=20)。如果直接用超出范围的坐标去访问数组,
 * 会越界崩溃。所以先夹回合法范围,只填图片内能放下的一部分。
 *
 * 为什么右边缘要 "x_0 + w - 1"?
 * 因为 w 是"格子的个数",而 x_1 是"最后一个格子的下标"。
 * 从 x_0 开始数 w 个格子,最后一个就是 x_0 + w - 1。
 * 减 1 是把"起点格子本身"算进去。举例:从 3 开始、宽 4,
 * 格子是 3,4,5,6,最后一个 6 = 3+4-1。 */
void layer_fill_rectangle(Layer layer, int x, int y, int w, int h, float value)
{
    /* assert: 断言。若条件为假(这里 w<=0)则程序直接报错退出。
     * 用于尽早发现"调用方用错了"的情况。Release 版会被自动移除。 */
    assert(w > 0);
    assert(h > 0);

    /* 四个边界都夹到图片内 */
    int x_0 = clamp_int(x,         0, WIDTH - 1);
    int y_0 = clamp_int(y,         0, HEIGHT - 1);
    int x_1 = clamp_int(x_0 + w - 1, 0, WIDTH - 1);   /* 右边界 = 左边 + 宽 - 1 */
    int y_1 = clamp_int(y_0 + h - 1, 0, HEIGHT - 1);  /* 下边界 = 上边 + 高 - 1 */

    /* 双重循环填满矩形内每个格子 */
    for (int y_i = y_0; y_i <= y_1; ++y_i)
        for (int x_i = x_0; x_i <= x_1; ++x_i)
            layer[y_i][x_i] = value;
}

/* 在图层上填充一个实心圆。圆心 (cx, cy),半径 r。
 * 判断:一个格子 (x_i, y_i) 离圆心的距离是否 <= r。
 * 勾股定理:距离平方 = dx^2 + dy^2,和 r^2 比较。
 * 用平方比较避免了开方(sqrt 较慢),这是图形学里的常用技巧。 */
void layer_fill_circle(Layer layer, int cx, int cy, int r, float value)
{
    assert(r > 0);

    /* 先夹出圆的外接正方形范围(只遍历这部分,省事) */
    int x_0 = clamp_int(cx - r, 0, WIDTH - 1);   /* 左边界 = 圆心 - 半径 */
    int y_0 = clamp_int(cy - r, 0, HEIGHT - 1);  /* 上边界 = 圆心 - 半径 */
    int x_1 = clamp_int(cx + r, 0, WIDTH - 1);   /* 右边界 = 圆心 + 半径 */
    int y_1 = clamp_int(cy + r, 0, HEIGHT - 1);  /* 下边界 = 圆心 + 半径 */

    const int r2 = r * r;   /* 半径的平方,提前算好(循环里反复用) */
    for (int y_i = y_0; y_i <= y_1; ++y_i) {
        for (int x_i = x_0; x_i <= x_1; ++x_i) {
            int dx = x_i - cx;          /* 该列离圆心的水平距离 */
            int dy = y_i - cy;          /* 该行离圆心的垂直距离 */
            if (dx * dx + dy * dy <= r2)   /* 距离平方 <= 半径平方,说明在圆内 */
                layer[y_i][x_i] = value;
        }
    }
}

/* 把图层导出为 PPM(P6) 图片,把训练时"生成/预测"的画面存成图片给人看。
 *
 * PPM 格式 = 文本头部 + 二进制像素数据。
 *   头部: "P6\n<宽> <高> 255\n"  (P6 声明是 RGB 彩色格式,255 指每通道最大 0~255)
 *   像素: 每个像素 3 个字节(红、绿、蓝)
 *
 * 这里把每个像素放大 PPM_SCALER 倍(50x50 变成 1250x1250),
 * 因为原图太小看不清。放大就是每个原像素画成一整块 PPM_SCALER x PPM_SCALER 的小方块。
 *
 * floorf(255*v)+0.5:把灰度值 0~1 映射到 0~255 的整数,并四舍五入。
 *   v=0 -> 0(黑), v=1 -> 255(白)。 */
int layer_save_as_ppm(Layer layer, const char *file_path)
{
    /* fopen 打开文件。"wb" = write 二进制(图片含非文本字节,必须用二进制模式)。
     * 失败返回 NULL(比如路径不存在、无权限)。 */
    FILE *f = fopen(file_path, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", file_path);
        return -1;
    }

    int w = WIDTH * PPM_SCALER;   /* 放大后的图片宽度 */
    int h = HEIGHT * PPM_SCALER;  /* 放大后的图片高度 */
    fprintf(f, "P6\n%d %d 255\n", w, h);   /* 写文件头 */

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            /* 把放大后的坐标映射回原图层坐标(整除) */
            float v = layer[y / PPM_SCALER][x / PPM_SCALER];
            unsigned char pixel[3] = {
                (unsigned char)(floorf(255.0f * v) + 0.5f),  /* 红通道 = 灰度 */
                0,                                            /* 绿通道 = 0 */
                0                                             /* 蓝通道 = 0 */
            };
            /* fwrite:把这 3 个字节写进文件 */
            fwrite(pixel, sizeof(pixel), 1, f);
        }
    }

    fclose(f);   /* 关闭文件,把缓冲区数据真正刷到磁盘 */
    return 0;
}

/* 把图层原封不动地写成二进制文件。
 * 用途:把"生成的样本"存下来。存的是 raw 二进制(不含任何头),
 * 加载时按同样顺序读回,就能还原一张完全相同的图。
 *
 * fopen "wb" 二进制写。fwrite(layer, sizeof(Layer), 1, f) 意思是:
 *   从 layer 地址,写 sizeof(Layer) 这么大、共 1 个元素到文件 f。
 * sizeof(Layer) 就是整个二维数组的字节数(50x50x4 = 10000 字节)。 */
int layer_save_as_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", file_path);
        return -1;
    }
    size_t written = fwrite(layer, sizeof(Layer), 1, f);
    fclose(f);
    return written == 1 ? 0 : -1;   /* 写成功 1 个元素返回 0,否则 -1 */
}

/* 从二进制文件读回一个图层(和 save 对称)。
 * fread 把文件里的字节读进 layer 内存,还原成图。 */
int layer_load_from_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "rb");   /* "rb" = read 二进制 */
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", file_path);
        return -1;
    }
    size_t read = fread(layer, sizeof(Layer), 1, f);
    fclose(f);
    return read == 1 ? 0 : -1;   /* 读到 1 个元素返回 0,否则 -1 */
}

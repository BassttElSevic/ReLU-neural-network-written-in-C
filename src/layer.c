#include "layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

int clamp_int(int x, int low, int high)
{
    if (x < low)   x = low;
    if (x > high)  x = high;
    return x;
}

void layer_clear(Layer layer, float value)
{
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            layer[y][x] = value;
}

/* 在图层上填充一个矩形区域。
 * (x, y) 为左上角,宽 w、高 h,所有落在图片内的格子都置为 value。
 * 注意:矩形可能超出图片边界,这里用 clamp_int 把坐标夹回 [0, WIDTH-1]/[0, HEIGHT-1]。
 */
void layer_fill_rectangle(Layer layer, int x, int y, int w, int h, float value)
{
    assert(w > 0);
    assert(h > 0);

    int x_0 = clamp_int(x,         0, WIDTH - 1);
    int y_0 = clamp_int(y,         0, HEIGHT - 1);
    int x_1 = clamp_int(x_0 + w - 1, 0, WIDTH - 1);
    int y_1 = clamp_int(y_0 + h - 1, 0, HEIGHT - 1);

    for (int y_i = y_0; y_i <= y_1; ++y_i)
        for (int x_i = x_0; x_i <= x_1; ++x_i)
            layer[y_i][x_i] = value;
}

/* 在图层上填充一个实心圆。圆心 (cx, cy),半径 r。
 * 用逐像素判断点到圆心距离是否 <= r 的方式填充。
 */
void layer_fill_circle(Layer layer, int cx, int cy, int r, float value)
{
    assert(r > 0);

    int x_0 = clamp_int(cx - r, 0, WIDTH - 1);
    int y_0 = clamp_int(cy - r, 0, HEIGHT - 1);
    int x_1 = clamp_int(cx + r, 0, WIDTH - 1);
    int y_1 = clamp_int(cy + r, 0, HEIGHT - 1);

    const int r2 = r * r;
    for (int y_i = y_0; y_i <= y_1; ++y_i) {
        for (int x_i = x_0; x_i <= x_1; ++x_i) {
            int dx = x_i - cx;
            int dy = y_i - cy;
            if (dx * dx + dy * dy <= r2)
                layer[y_i][x_i] = value;
        }
    }
}

/* 把图层导出为 PPM(P6) 图片。PPM 由文本头部 + 二进制像素组成。
 * 这里把每个像素放大 PPM_SCALER 倍,方便肉眼查看。
 */
int layer_save_as_ppm(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", file_path);
        return -1;
    }

    int w = WIDTH * PPM_SCALER;
    int h = HEIGHT * PPM_SCALER;
    fprintf(f, "P6\n%d %d 255\n", w, h);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = layer[y / PPM_SCALER][x / PPM_SCALER];
            unsigned char pixel[3] = {
                (unsigned char)(floorf(255.0f * v) + 0.5f),
                0,
                0
            };
            fwrite(pixel, sizeof(pixel), 1, f);
        }
    }

    fclose(f);
    return 0;
}

/* 把图层原封不动地写成二进制(训练/加载数据用,不含任何格式头)。 */
int layer_save_as_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "wb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", file_path);
        return -1;
    }
    size_t written = fwrite(layer, sizeof(Layer), 1, f);
    fclose(f);
    return written == 1 ? 0 : -1;
}

int layer_load_from_bin(Layer layer, const char *file_path)
{
    FILE *f = fopen(file_path, "rb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: could not open file %s\n", file_path);
        return -1;
    }
    size_t read = fread(layer, sizeof(Layer), 1, f);
    fclose(f);
    return read == 1 ? 0 : -1;
}

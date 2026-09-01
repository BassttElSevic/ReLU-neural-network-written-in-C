#include "dataset.h"
#include "rng.h"
#include "network.h"
#include "layer.h"
#include <assert.h>

void dataset_flatten(Layer layer, float *flat)
{
    for (int y = 0; y < HEIGHT; ++y)
        for (int x = 0; x < WIDTH; ++x)
            flat[y * WIDTH + x] = layer[y][x];
}

void dataset_random_sample(Layer layer, float *flat, int *label)
{
    /* 等概率随机选择画矩形(0)或圆形(1) */
    if (rand_range(0, 2) == 0) {
        /* --- 随机矩形 --- */
        layer_clear(layer, 0.0f);

        int x = rand_range(0, WIDTH);
        int y = rand_range(0, HEIGHT);

        int w = WIDTH - x;
        if (w < 2) w = 2;
        w = rand_range(1, w + 1);

        int h = HEIGHT - y;
        if (h < 2) h = 2;
        h = rand_range(1, h + 1);

        layer_fill_rectangle(layer, x, y, w, h, 1.0f);
        *label = 0;
    } else {
        /* --- 随机圆形 --- */
        layer_clear(layer, 0.0f);

        int cx = rand_range(0, WIDTH);
        int cy = rand_range(0, HEIGHT);

        /* 半径上限取到四个边界的最小距离,保证圆不越界 */
        int r_max = cx;
        if (cy < r_max)            r_max = cy;
        if (WIDTH - cx < r_max)    r_max = WIDTH - cx;
        if (HEIGHT - cy < r_max)   r_max = HEIGHT - cy;
        if (r_max < 2)             r_max = 2;

        int r = rand_range(2, r_max + 1);
        layer_fill_circle(layer, cx, cy, r, 1.0f);
        *label = 1;
    }

    dataset_flatten(layer, flat);
}

#ifndef LAYER_H
#define LAYER_H

/* 定义图层:一张 HEIGHT x WIDTH 的灰度图,每个像素取值 0.0(背景)到 1.0(前景)。 */
#define WIDTH  50
#define HEIGHT 50
#define PPM_SCALER 25   /* PPM 导出时的放大倍数,便于肉眼查看 */

typedef float Layer[HEIGHT][WIDTH];

int  clamp_int(int x, int low, int high);

void layer_clear(Layer layer, float value);
void layer_fill_rectangle(Layer layer, int x, int y, int w, int h, float value);
void layer_fill_circle(Layer layer, int cx, int cy, int r, float value);

int  layer_save_as_ppm(Layer layer, const char *file_path);
int  layer_save_as_bin(Layer layer, const char *file_path);
int  layer_load_from_bin(Layer layer, const char *file_path);

#endif /* LAYER_H */

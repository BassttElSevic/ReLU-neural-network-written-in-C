#ifndef LAYER_H
#define LAYER_H

/* =====================================================================
 * 图层(Layer)模块:一张灰度图 + 在图上画形状 + 存取。
 * 这是整个项目"数据从哪来"的基础。
 * ===================================================================== */

/* 图层尺寸:一张 50x50 的图,共 2500 个像素。
 * 用宏定义而不是直接写数字,方便以后改大小,也便于维护。 */
#define WIDTH  50
#define HEIGHT 50

/* PPM 导出时的放大倍数。原图 50x50 太小看不清,
 * 导出图片时放大到 50x50 的 PPM_SCALER 倍。 */
#define PPM_SCALER 25

/* Layer 是一个"二维数组类型":HEIGHT 行 x WIDTH 列的 float。
 *
 * 注意!这里 typedef float Layer[HEIGHT][WIDTH]; 定义的是【类型】,
 * 不是变量。之后写 "Layer x" 就相当于声明了一个 50x50 的 float 数组。
 *
 * 每个像素取值 0.0(背景/黑)到 1.0(前景/白)。
 * 这个数组直接用"行列"访问:Layer[y][x]。 */
typedef float Layer[HEIGHT][WIDTH];

/* 把整数 x 限制在 [low, high] 区间,防止越界 */
int  clamp_int(int x, int low, int high);

/* 把整个图层填成同一个值(0.0=黑背景,1.0=白) */
void layer_clear(Layer layer, float value);

/* 画一个矩形:左上角(x,y),宽 w,高 h,内部填 value */
void layer_fill_rectangle(Layer layer, int x, int y, int w, int h, float value);

/* 画一个实心圆:圆心(cx,cy),半径 r,内部填 value */
void layer_fill_circle(Layer layer, int cx, int cy, int r, float value);

/* 把图层存成 PPM 图片(给人看),成功返回 0,失败返回 -1 */
int  layer_save_as_ppm(Layer layer, const char *file_path);

/* 把图层存成二进制(存数据给程序自己读),成功返回 0,失败返回 -1 */
int  layer_save_as_bin(Layer layer, const char *file_path);

/* 从二进制文件读回图层,成功返回 0,失败返回 -1 */
int  layer_load_from_bin(Layer layer, const char *file_path);

#endif /* LAYER_H */

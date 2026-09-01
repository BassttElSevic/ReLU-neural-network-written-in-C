#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
// #include <errno.h>
#include <math.h>
#include <limits.h>

#define WIDTH 50
#define HEIGHT 50
#define PPM_SCALER 25
#define SAMPLE_SIZE 10

typedef float Layer[HEIGHT][WIDTH];

// inline 内联是给编译器的建议，优化之后减少函数调用开销
static inline int clamp_int(int x, int low, int high)
{
	// 把坐标限制在图片范围的工具
	if (x < low) {
		x = low;
	} 
	if (x > high) { 
		x = high;
	}
	return x;
}
	
// 举个例子，假设我起点是3,宽度是4,我想要整填充3,4,5,6的话，我终点就必须是，起点加宽度再减一，这样才不会出去
// 这里的x和y就是矩形的起始点
void layer_fill_rectangle(Layer layer, int x, int y, int w, int h, float value)
{
	assert(w > 0);
	assert(h > 0);
	int x_0 = clamp_int(x, 0, WIDTH-1);
	int y_0 = clamp_int(x, 0, HEIGHT-1);
	int x_1 = clamp_int(x_0 + w -1, 0, WIDTH-1);
	int y_1 = clamp_int(y_0 + w -1, 0, HEIGHT-1);
	for (int y_i = y_0; y_i <= y_1; ++y_i) {
		for (int x_i = x_0; x_i <= x_1; ++x_i) {
			 layer[y_i][x_i] = value;
		} 
	}
}
// 当然，只有矩形的还是不行，我们还需要画圆形的
void layer_fill_circle(Layer layer, int circle_x, int circle_y, int r, float value)
{
	assert(r > 0);
	int x_0 = clamp_int(circle_x - r, 0, WIDTH - 1);
	int y_0 = clamp_int(circle_y - r, 0, HEIGHT - 1);
	int x_1 = clamp_int(circle_x + r, 0, WIDTH - 1);
	int y_1 = clamp_int(circle_y + r, 0, HEIGHT - 1);
	for(int y_i = y_0; y_i <= y_1; ++y_i) {
		for(int x_i = x_0; x_i <= x_1; ++x_i) {
			int delta_x = x_i - circle_x;
			int delta_y = y_i - circle_y;
			const int squre_of_r = r * r;
			if( ((delta_x * delta_x) + (delta_y * delta_y)) <= (squre_of_r) ) {
				layer[y_i][x_i] = value;
			}
		}
	}
}
// .ppm格式转化，.ppm由前面的字节头还有后面的二进制部分组成
void layer_save_as_ppm(Layer layer, const char *file_path)
{
	FILE *f = fopen(file_path, "wb");
	if (f == NULL) {
		fprintf(stderr, "ERROR: Could not open file %s: %m\n",file_path);
		exit(1);
	}
	fprintf(f, "P6\n%d %d 255\n",WIDTH * PPM_SCALER, HEIGHT * PPM_SCALER);
	for(int y = 0; y < HEIGHT * PPM_SCALER; ++y) {
		for(int x =0; x < WIDTH * PPM_SCALER; ++x) {
			char pixel[3] = {
				(char) floorf(255 * layer[y / PPM_SCALER ][x /PPM_SCALER ]), // 红通道
				0, // 绿通道
				0  // 蓝通道
			};
			fwrite(pixel, sizeof(pixel), 1, f);
		}
	}
	fclose(f);
}
// 把layer保留成二进制
void layer_save_as_bin(Layer layer, const char *file_path)
{
	FILE *f = fopen(file_path, "wb");
	if (f == NULL) {
		fprintf(stderr, "ERROR: Could not open file %s: %m", file_path);
		exit(1);
	}
	fwrite(layer, sizeof(Layer), 1, f);
	fclose(f);
}
// 当然，也可以从二进制中加载
void layer_load_from_bin(Layer layer, const char *file_path)
{
	assert(0 && "TO-DO: layer_load_from_bin is not implemented yet! ");
}
// 经典的前馈神经元
float neuron_forward(Layer inputs, Layer weights)
{
	float output = 0.0f;
	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			output += inputs[y][x] * weights[y][x];
		}
	}
	return output;
}

// 一个我们需要随机用的
int rand_range(int low, int high)
{
	assert(low < high);
	return rand() % (high - low) + low;
}

static Layer inputs;
static Layer weights;

void layer_random_rectangle(Layer layer)
{
	layer_fill_rectangle(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
	int x = rand_range(0, WIDTH);
	int y = rand_range(0, HEIGHT);
	
	int w = WIDTH - x;
	if (w < 2) {
		w = 2;
	}
	w = rand_range(1,w);

	int h = HEIGHT - y;
	if (y < 2) {
		y = 2;
	}
	y = rand_range(1,y);

	layer_fill_rectangle(layer, x, y, w, h, 1.0f);
}

void layer_random_circle(Layer layer)
{
	layer_fill_rectangle(layer, 0, 0, WIDTH, HEIGHT, 0.0f);
	int circle_x = rand_range(0, WIDTH);
	int circle_y = rand_range(0, HEIGHT);
	int r = INT_MAX;
	if (r > circle_x) {
		r = circle_x;
	}
	if (r > circle_y) {
		r = circle_y;
	}
	if (r > WIDTH - circle_x) {
		r = WIDTH - circle_x;
	}
	if (r > HEIGHT - circle_y) {
		r = HEIGHT - circle_y;
	}
	r = rand_range(1,r);
	layer_fill_circle(layer, circle_x, circle_y, r, 1.0f);
}

int main ()
{
//	printf("Hello World!\n");
//	float output = neuron_forward(inputs,weights);
//	printf("output = %f\n",output);
//	layer_fill_rectangle(inputs, 0, 0, WIDTH/2, HEIGHT/2, 1.0f);//
	// layer_fill_circle(inputs, WIDTH/2, HEIGHT/2, WIDTH/2, 1.0f);
	// layer_save_as_ppm(inputs,"inputs.ppm");
	//layer_fill_circle(inputs,WIDTH / 2, HEIGHT / 2, WIDTH / 2, 1.0f);
	//layer_save_as_ppm(inputs, "inputs.ppm");
	char file_path[256];
	for (int i = 0; i < SAMPLE_SIZE; ++i) {
		printf("[INFO]: generating circle %d\n", i);
		// 先打印目前的信息，然后再画出来

		layer_random_circle(inputs);

		snprintf(file_path, sizeof(file_path), "circle-%02d.bin", i);
		layer_save_as_bin(inputs, file_path);
		snprintf(file_path, sizeof(file_path), "circle-%02d.ppm", i);
		layer_save_as_ppm(inputs, file_path);
	}
	return 0;
}  

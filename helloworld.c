#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
// #include <errno.h>
#include <math.h>

#define WIDTH 20
#define HEIGHT 20
#define PPM_SCALAR 10

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

void layer_save_as_ppm(Layer layer, const char *file_path)
{
	FILE *f = fopen(file_path, "wb");
	if (f == NULL) {
		fprintf(stderr, "ERROR: Could not open file %s: %m\n",file_path);
		exit(1);
	}
	fprintf(f, "P6\n%d %d 255\n",WIDTH, HEIGHT);
	for(int y = 0; y < HEIGHT; ++y) {
		for(int x =0; x < WIDTH; ++x) {
			char pixel[3] = {
				(char) floorf(255 * layer[y][x]), // 红通道
				0, // 绿通道
				0  // 蓝通道
			};
			fwrite(pixel, sizeof(pixel), 1, f);
		}
	}
	fclose(f);
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

static Layer inputs;
static Layer weights;
	
int main ()
{
//	printf("Hello World!\n");
//	float output = neuron_forward(inputs,weights);
//	printf("output = %f\n",output);
	layer_fill_rectangle(inputs, 0, 0, WIDTH/2, HEIGHT/2, 1.0f);
	layer_save_as_ppm(inputs,"inputs.ppm");
	return 0;
}  

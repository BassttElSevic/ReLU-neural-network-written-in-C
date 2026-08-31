#include <stdio.h>
#include <assert.h>

#define WIDTH 20
#define HEIGHT 20

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
	printf("Hello World!\n");
	float output = neuron_forward(inputs,weights);
	printf("output = %f\n",output);
	return 0;
}  

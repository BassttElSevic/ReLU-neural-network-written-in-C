#include "stdio.h"

#define WIDTH 20
#define HEIGHT 20

float feed_forward(float inputs[HEIGHT][WIDTH], float weights[HEIGHT][WIDTH])
{
	float output = 0.0f;
	for (int y = 0; y < HEIGHT; ++y) {
		for (int x = 0; x < WIDTH; ++x) {
			output += inputs[y][x] * weights[y][x];
		}
	}
	return output;
}

int main ()
{
	printf("Hello World!\n");
	
	return 0;
}  

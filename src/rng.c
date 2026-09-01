#include "rng.h"
#include <stdlib.h>

void rng_seed(int seed)
{
    srand((unsigned int)seed);
}

int rand_range(int low, int high)
{
    /* 前提:low < high;返回 [low, high) 内的整数 */
    return rand() % (high - low) + low;
}

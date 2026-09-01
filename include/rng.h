#ifndef RNG_H
#define RNG_H

void rng_seed(int seed);
int  rand_range(int low, int high);  /* 返回 [low, high) 内的随机整数 */

#endif /* RNG_H */

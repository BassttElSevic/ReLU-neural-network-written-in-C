#include "rng.h"
#include <stdlib.h>

/* 设置随机数种子。
 *
 * 计算机的随机是"伪随机":rand() 看起来随机,其实是一串固定的数列。
 * 种子(seed)决定从数列的哪个位置开始。
 *   - 不设种子(或每次用系统时间):每次运行结果不同,训练结果会有差异。
 *   - 设固定种子(比如 12345):每次都得到同一串随机数,训练结果可复现。
 * 这就是为什么 main 里能通过参数传种子。
 * */
void rng_seed(int seed)
{
    srand((unsigned int)seed);
}

/* 返回 [low, high) 内的随机整数(注意:不含 high)。
 * rand() % (high - low):把随机数对"区间长度"取余,得到 0 ~ (high-low-1)。
 * 再加 low,就落在 [low, high) 范围内。
 *
 * 前提:low < high,否则取余会出错。 */
int rand_range(int low, int high)
{
    /* rand() % (high - low) 这个取余法有个小缺点:不是完全均匀,
     * 但对我们生成训练样本来说足够用了。 */
    return rand() % (high - low) + low;
}

#ifndef DATASET_H
#define DATASET_H

#include "layer.h"

/* 生成一个随机样本。
 * 随机画一个矩形或一个圆形,写入 layer,并把对应类别写入 *label。
 *   label = 0 表示矩形,label = 1 表示圆形。
 * 同时把 layer 拉平成 float 数组写入 flat(长度 INPUT_SIZE)。
 */
void dataset_random_sample(Layer layer, float *flat, int *label);

/* 把一张已经画好的图层拉平为输入向量 */ 
void dataset_flatten(Layer layer, float *flat);

#endif /* DATASET_H */

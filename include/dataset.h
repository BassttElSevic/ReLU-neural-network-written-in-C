#ifndef DATASET_H
#define DATASET_H

#include "layer.h"

/* =====================================================================
 * 数据集模块:随机生成训练样本(矩形或圆形)。
 * 因为用程序就能生成样本,所以不需要人工标注大量图片。
 * ===================================================================== */

/* 生成一个随机样本。
 * 随机画一个矩形或一个圆形,写入 layer;把类别写入 *label(0=矩形,1=圆形);
 * 同时把 layer 拉平成一维数组写入 flat(长度 INPUT_SIZE)。 */
void dataset_random_sample(Layer layer, float *flat, int *label);

/* 把一张已经画好的图层拉平为一维输入向量(50x50 -> 2500) */
void dataset_flatten(Layer layer, float *flat);

#endif /* DATASET_H */

#include "yolov3_layers.h"

float coeff_buffer[COEFF_NUM];
float input_img[CHANNELS * WIDTH * HEIGHT];

int main() {
  readCoeff(COEFF_FILENAME, coeff_buffer);
  readImg(IN_FILENAME, input_img);

  // Reading coeffs for first conv layer
  int layer_0_filter_num = 16;
  float biases[layer_0_filter_num];
  float bn_weights[layer_0_filter_num];
  float bn_running_mean[layer_0_filter_num];
  float bn_running_var[layer_0_filter_num];
  float conv_weight[3 * 3 * 3 * layer_0_filter_num];

  int i = 5;
  memcpy(biases, coeff_buffer + i, layer_0_filter_num * sizeof(float));
  i += layer_0_filter_num;

  memcpy(bn_weights, coeff_buffer + i, layer_0_filter_num * sizeof(float));
  i += layer_0_filter_num;

  memcpy(bn_running_mean, coeff_buffer + i, layer_0_filter_num * sizeof(float));
  i += layer_0_filter_num;

  memcpy(bn_running_var, coeff_buffer + i, layer_0_filter_num * sizeof(float));
  i += layer_0_filter_num;

  memcpy(conv_weight, coeff_buffer + i,
         3 * 3 * 3 * layer_0_filter_num * sizeof(float));
  i += 3 * 3 * 3 * layer_0_filter_num;

  float padded_buffer[CHANNELS * (WIDTH + 2) * (HEIGHT + 2)];
  image_padding(input_img, padded_buffer);

  return 0;
}
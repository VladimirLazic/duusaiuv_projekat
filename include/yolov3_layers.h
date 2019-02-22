#ifndef YOLOV3_LAYERS_H
#define YOLOV3_LAYERS_H

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h> /* atof */
#include <string>

#define FILEWRITE_TASK_STACK_SIZE (32 * KB)
#define NULLSRC_TIMER_PERIOD_FW (33U)
#define WIDTH (416U)
#define HEIGHT (416U)
#define NUM_OUTPUT (10U)
#define FRAME_SIZE (WIDTH * HEIGHT)
#define NUM_WEIGHTS (WIDTH * HEIGHT * NUM_OUTPUT)
#define NUM_BIAS (10U)
#define IN_FILENAME "../test-data/giraffe.rgb"
#define COEFF_FILENAME "../yolov3-tiny.weights"

#define COEFF_NUM (8858739U)
#define CHANNELS (3U)

typedef struct {
  float *biases;
  float *bn_weights = NULL;
  float *bn_running_mean = NULL;
  float *bn_running_var = NULL;
  float *conv_weight;
  bool batch_normalization = false;
} conv_configuration;

static void batch_normalization(float *input_buffer, float *output_buffer,
                                conv_configuration cfg);

static void conv2d(float *input_buffer, float *output_buffer,
                   conv_configuration cfg);

void conv(float *input_buffer, float *output_buffer, conv_configuration cfg);

void readCoeff(std::string file_path, float *coeff_buffer) {
  FILE *file;
  size_t nread;

  file = fopen(file_path.c_str(), "r");
  if (file) {
    nread = fread(coeff_buffer, sizeof(float), COEFF_NUM * sizeof(float), file);
    fclose(file);
  }
}

void readImg(std::string file_path, float *input_img) {
  FILE *file;
  size_t nread;

  file = fopen(file_path.c_str(), "r");
  if (file) {
    nread = fread(input_img, sizeof(float),
                  CHANNELS * WIDTH * HEIGHT * sizeof(float), file);
    fclose(file);
  }
}

#endif // YOLOV3_LAYERS_H
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

typedef struct
{
    int filter_num;

    float* biases;
    float* bn_weights = NULL;
    float* bn_running_mean = NULL;
    float* bn_running_var = NULL;
    float* conv_weight;

    bool batch_normalization = false;

    int input_width;
    int input_height;

    int kernel_width;
    int kernel_height;
    int kernel_depth;

    int output_width;
    int output_height;
    int output_depth;

} conv_configuration;

static void image_padding(float* input_buffer, float* output_buffer)
{

    memset(output_buffer, 0, (WIDTH + 2) * (HEIGHT + 2) * sizeof(float));

    for (int i = 0; i < HEIGHT; i++)
    {
        memcpy(output_buffer + (i + 1) * (WIDTH + 2), input_buffer + i * WIDTH,
               WIDTH * sizeof(float));
    }
}

static void batch_normalization(float* input_buffer, float* output_buffer,
                                conv_configuration cfg);

void conv2d(float* input_buffer, float* output_buffer, conv_configuration cfg)
{
    int input_width = cfg.input_width;
    int input_height = cfg.input_height;

    int new_image_width = cfg.output_width;
    int new_image_height = cfg.output_height;

    float* layer_output_buffer =
        new float[16 * 3 * new_image_width * new_image_height];
    memset(layer_output_buffer, 0,
           16 * 3 * new_image_width * new_image_height * sizeof(float));

    for (int i = 0; i < cfg.filter_num; i++)
    {
        float kernels[3 * 3 * 3];
        memcpy(kernels, cfg.conv_weight + i * 3 * 3 * 3,
               3 * 3 * 3 * sizeof(float));

        float output_img[3 * new_image_width * new_image_height];
        memset(output_img, 0,
               3 * new_image_width * new_image_height * sizeof(float));

        for (int j = 0; j < 3; j++)
        {
            // Separate kernel accoriding to channels
            float kernel[3 * 3];
            memcpy(kernel, kernel + j * 3 * 3, 3 * 3 * sizeof(float));

            // Sepparate channels and perform padding
            float channel[input_width * input_height];
            mempcpy(channel, input_buffer + j * input_width * input_height,
                    input_width * input_height * sizeof(float));

            float padded_channel[(input_width + 2) * (input_height + 2)];
            image_padding(channel, padded_channel);

            float output_channel[new_image_width * new_image_height];
            memset(output_channel, 0,
                   new_image_width * new_image_height * sizeof(float));
            // Apply kernel

            for (int ii = 3 / 2; ii < (input_width + 2) - 3 / 2; ++ii)
            {
                for (int jj = 3 / 2; jj < (input_height + 2) - 3 / 2; ++jj)
                {
                    float sum = 0;

                    for (int x = -3 / 2; x <= 3 / 2; ++x)
                    {
                        for (int y = -3 / 2; y <= 3 / 2; ++y)
                        {
                            float data =
                                padded_channel[(jj + y) * (input_width + 2) +
                                               (ii + x)];

                            float coeff =
                                kernel[(y + 3 / 2) * (input_width + 2) +
                                       (x + 3 / 2)];

                            sum += data * coeff;
                        }
                    }
                    output_channel[jj * new_image_width + ii] = sum;
                }
            }
            // Combine the channels together
            memcpy(output_img + j * new_image_width * new_image_height,
                   output_channel,
                   new_image_width * new_image_height * sizeof(float));
        }

        memcpy(layer_output_buffer + i * 3 * new_image_width * new_image_height,
               output_img,
               3 * new_image_width * new_image_height * sizeof(float));
    }

    if (cfg.batch_normalization)
    {
        // batch_normalization(layer_output_buffer, output_buffer, cfg);
    }
    else
    {
        memcpy(output_buffer, layer_output_buffer,
               16 * 3 * new_image_width * new_image_height * sizeof(float));
    }
}

void readCoeff(std::string file_path, float* coeff_buffer)
{
    FILE* file;
    size_t nread;

    file = fopen(file_path.c_str(), "r");
    if (file)
    {
        nread =
            fread(coeff_buffer, sizeof(float), COEFF_NUM * sizeof(float), file);
        fclose(file);
    }
}

void readImg(std::string file_path, float* input_img)
{
    FILE* file;
    size_t nread;

    file = fopen(file_path.c_str(), "r");
    if (file)
    {
        nread = fread(input_img, sizeof(float),
                      CHANNELS * WIDTH * HEIGHT * sizeof(float), file);
        fclose(file);
    }
}

#endif // YOLOV3_LAYERS_H

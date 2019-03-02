#include "yolov3_layers.h"

float coeff_buffer[COEFF_NUM];
float input_img[CHANNELS * WIDTH * HEIGHT];

int main()
{
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

    memcpy(bn_running_mean, coeff_buffer + i,
           layer_0_filter_num * sizeof(float));
    i += layer_0_filter_num;

    memcpy(bn_running_var, coeff_buffer + i,
           layer_0_filter_num * sizeof(float));
    i += layer_0_filter_num;

    memcpy(conv_weight, coeff_buffer + i,
           3 * 3 * 3 * layer_0_filter_num * sizeof(float));
    i += 3 * 3 * 3 * layer_0_filter_num;

    conv_configuration layer1;

    layer1.filter_num = layer_0_filter_num;

    layer1.biases = biases;
    layer1.bn_weights = bn_weights;
    layer1.bn_running_mean = bn_running_mean;
    layer1.bn_running_var = bn_running_var;
    layer1.conv_weight = conv_weight;

    layer1.batch_normalization = false;

    layer1.input_width = WIDTH;
    layer1.input_height = HEIGHT;

    layer1.kernel_height = 3;
    layer1.kernel_width = 3;
    layer1.kernel_depth = 3;

    layer1.output_width = (layer1.input_width + 2) - layer1.kernel_width + 1;
    layer1.output_height = (layer1.input_height + 2) - layer1.kernel_height + 1;
    layer1.output_depth = 16;

    float* layer1_output =
        new float[layer1.output_depth * 3 * layer1.output_width *
                  layer1.output_height];

    conv2d(input_img, layer1_output, layer1);

    for (int i = 0; i < 1000; i++)
    {
        std::cout << layer1_output[i] << std::endl;
    }

    return 0;
}

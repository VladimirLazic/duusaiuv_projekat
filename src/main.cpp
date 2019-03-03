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

    layer1.batch_normalization = true;

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

    /*conv(input_img, layer1_output, layer1);

    // Testing convolution
    for (int i = 0; i < 1000; i++)
    {
        std::cout << layer1_output[i] << std::endl;
    }*/

    maxpool_conf maxpool_cfg;

    maxpool_cfg.size = 2;
    maxpool_cfg.stride = 2;

    maxpool_cfg.input_width = WIDTH;
    maxpool_cfg.input_height = HEIGHT;

    maxpool_cfg.output_width = WIDTH / 2;
    maxpool_cfg.output_height = HEIGHT / 2;

    float* output_maxpool = new float[(CHANNELS * WIDTH * HEIGHT) / 4];

    maxpool(input_img, output_maxpool, maxpool_cfg);

    std::cout << "MAX POOL " << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << output_maxpool[i] << std::endl;
    }
    std::cout << "\n\n\n";

    upsample_conf upsample_cfg;

    upsample_cfg.stride = 2;

    upsample_cfg.input_width = WIDTH;
    upsample_cfg.input_height = HEIGHT;

    upsample_cfg.output_width = WIDTH * 2;
    upsample_cfg.output_height = HEIGHT * 2;

    float* output_upsample = new float[(CHANNELS * WIDTH * HEIGHT) * 4];
    upsample(input_img, output_upsample, upsample_cfg);

    std::cout << "UPSAMPLE" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << output_upsample[i] << std::endl;
    }

    return 0;
}

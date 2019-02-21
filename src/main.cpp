#include <iostream>
#include <string>
#include <stdlib.h>     /* atof */
#include <sstream>
#include <fstream>
#include <cstring>

#define FILEWRITE_TASK_STACK_SIZE      (32 * KB)
#define NULLSRC_TIMER_PERIOD_FW        (33U)
#define WIDTH                          (416U)
#define HEIGHT                         (416U)
#define NUM_OUTPUT                     (10U)
#define FRAME_SIZE                     (WIDTH * HEIGHT)
#define NUM_WEIGHTS                    (WIDTH * HEIGHT * NUM_OUTPUT)
#define NUM_BIAS                       (10U)
#define IN_FILENAME                    "../test-data/giraffe.rgb"
#define COEFF_FILENAME                 "../yolov3-tiny.weights"

#define COEFF_NUM                      (8858739U) 
#define CHANNELS                       (3U) 

float coeff_buffer[COEFF_NUM];
float input_img[CHANNELS * WIDTH * HEIGHT];

void readCoeff(std::string file_path, float* coeff_buffer) {
    FILE *file;
    size_t nread;
    

    file = fopen(file_path.c_str(), "r");
    if (file) {
        nread = fread(coeff_buffer, sizeof(float), COEFF_NUM * sizeof(float), file);    
        fclose(file);
    }
}

void readImg(std::string file_path, float* input_img) {
    FILE* file;
    size_t nread;

    file = fopen(file_path.c_str(), "r");
    if(file) {
        nread = fread(input_img, sizeof(float), CHANNELS * WIDTH * HEIGHT * sizeof(float), file);
        fclose(file);  
    }
}



int main() {

    readCoeff(COEFF_FILENAME, coeff_buffer);   
    readImg(IN_FILENAME, input_img);

    for(int i = 0; i < CHANNELS * WIDTH * HEIGHT; i++) {
        std::cout << input_img[i] << std::endl;
    }

    return 0;
}
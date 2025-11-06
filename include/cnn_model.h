#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include "matrix.h"
#include "matrix_cnn.h"

typedef struct {
    // convolution layer
    double** conv_kernel;
    double** d_conv_kernel;
    int k_size;

    // dense layer 
    double** dense_weights;
    double** d_dense_weights;
    double* dense_bias;
    double* d_dense_bias;
    int input_size;
    int output_size;

    // intermediate buffers
    double** conv_out;
    double** pool_out;
    int pool_out_size;
    int** pool_mask;

    double* flatten;
    double* dense_out;
    double* d_dense_out;
    double* softmax_out;
} CNN;

CNN* init_cnn();
double forward_cnn(CNN *net, double** input, int label);
void backward_cnn(CNN *net, double** input);
void update_params(CNN *net, double lr);
void free_cnn(CNN* net);

#endif

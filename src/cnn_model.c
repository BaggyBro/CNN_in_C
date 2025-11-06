#include "../include/cnn_model.h"
#include "../include/matrix.h"
#include "../include/matrix_cnn.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

CNN* init_cnn() {
    CNN* net = (CNN*)malloc(sizeof(CNN));

    // conv layer 3x3 kernel
    net->k_size = 3;
    net->conv_kernel = allocate_matrix(net->k_size, net->k_size);
    net->d_conv_kernel = allocate_matrix(net->k_size, net->k_size);

    for (int i = 0; i < net->k_size; i++) {
        for (int j = 0; j < net->k_size; j++) {
            net->conv_kernel[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }

    int conv_out_size = 28 - net->k_size + 1;
    int pool_size = 2;
    int pool_out_size = conv_out_size / pool_size;

    net->input_size = pool_out_size * pool_out_size;
    net->output_size = 10;

    net->dense_weights = allocate_matrix(net->output_size, net->input_size);
    net->d_dense_weights = allocate_matrix(net->output_size, net->input_size);
    net->dense_bias = (double*)malloc(net->output_size * sizeof(double));
    net->d_dense_bias = (double*)malloc(net->output_size * sizeof(double));

    for (int i = 0; i < net->output_size; i++) {
        net->dense_bias[i] = 0.0;
        for (int j = 0; j < net->input_size; j++) {
            net->dense_weights[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.1;
        }
    }

    return net;
}

double forward_cnn(CNN* net, double** input, int label) {
    int conv_out_size = 28 - net->k_size + 1;
    int pool_size = 2;
    int pool_out_size = conv_out_size / pool_size;
    net->pool_out_size = pool_out_size;

    net->conv_out = convolve_forward(input, 28, net->conv_kernel, net->k_size, &conv_out_size);
    relu_forward(net->conv_out, conv_out_size, conv_out_size);

    int **mask;
    net->pool_out = maxpool_forward(net->conv_out, conv_out_size, pool_size, &pool_out_size, &mask);
    net->pool_mask = mask;

    net->flatten = flatten_matrix(net->pool_out, pool_out_size, pool_out_size);

    net->dense_out = dense_forward(net->flatten, net->input_size,
                                   net->dense_weights, net->dense_bias, net->output_size);

    double* d_logits = (double*)malloc(net->output_size * sizeof(double));
    double loss = softmax_cross_entropy_loss_and_grad(net->dense_out, net->output_size, label, d_logits);

    net->d_dense_out = d_logits;
    net->softmax_out = softmax_forward(net->dense_out, net->output_size);

    return loss;
}

void backward_cnn(CNN* net, double** input) {
    double* d_input_flat = (double*)malloc(net->input_size * sizeof(double));

    dense_backward(net->d_dense_out, net->flatten, net->input_size,
                   net->dense_weights, net->output_size,
                   d_input_flat, net->d_dense_weights, net->d_dense_bias);

    double** d_pool = allocate_matrix(net->pool_out_size, net->pool_out_size);
    for (int i = 0; i < net->pool_out_size; i++) {
        for (int j = 0; j < net->pool_out_size; j++) {
            d_pool[i][j] = d_input_flat[i * net->pool_out_size + j];
        }
    }

    double** d_conv_out = allocate_matrix(28 - net->k_size + 1, 28 - net->k_size + 1);
    maxpool_backward(d_pool, net->pool_out_size, 2, 28 - net->k_size + 1, &(net->pool_mask), d_conv_out);
    relu_backward(net->conv_out, d_conv_out, 28 - net->k_size + 1, 28 - net->k_size + 1);

    double** d_input_conv = allocate_matrix(28, 28);
    convolve_backward(d_conv_out, 28 - net->k_size + 1, input, 28,
                      net->conv_kernel, net->k_size,
                      d_input_conv, net->d_conv_kernel);

    free(d_input_flat);
    free_matrix(d_pool, net->pool_out_size);
    free_matrix(d_conv_out, 28 - net->k_size + 1);
    free_matrix(d_input_conv, 28);
}

void update_params(CNN* net, double lr) {
    for (int i = 0; i < net->output_size; i++) {
        net->dense_bias[i] -= lr * net->d_dense_bias[i];
        for (int j = 0; j < net->input_size; j++) {
            net->dense_weights[i][j] -= lr * net->d_dense_weights[i][j];
        }
    }

    for (int i = 0; i < net->k_size; i++) {
        for (int j = 0; j < net->k_size; j++) {
            net->conv_kernel[i][j] -= lr * net->d_conv_kernel[i][j];
        }
    }
}

void free_cnn(CNN* net) {
    free_matrix(net->conv_kernel, net->k_size);
    free_matrix(net->d_conv_kernel, net->k_size);

    free_matrix(net->dense_weights, net->output_size);
    free_matrix(net->d_dense_weights, net->output_size);

    free_matrix(net->conv_out, 28 - net->k_size + 1);
    free_matrix(net->pool_out, net->pool_out_size);

    free(net->dense_bias);
    free(net->d_dense_bias);

    free(net->flatten);
    free(net->dense_out);
    free(net->softmax_out);
    free(net->d_dense_out);

    if (net->pool_mask) {
        for (int i = 0; i < net->pool_out_size; i++)
            free(net->pool_mask[i]);
        free(net->pool_mask);
    }

    free(net);
}
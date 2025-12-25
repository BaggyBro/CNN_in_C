#ifndef CNN_MODEL_H
#define CNN_MODEL_H

typedef struct {
    int k_size;
    int input_size;
    int output_size;
    int pool_out_size;

    double **conv_kernel;
    double **d_conv_kernel;

    double **dense_weights;
    double **d_dense_weights;
    double *dense_bias;
    double *d_dense_bias;

    double **conv_z;
    double **conv_out;
    double **pool_out;
    int **pool_mask;

    double *flatten;
    double *dense_out;
    double *softmax_out;
    double *d_dense_out;
} CNN;

CNN* init_cnn();
double forward_cnn(CNN* net, double** input, int label);
void backward_cnn(CNN* net, double** input);
void update_params(CNN* net, double lr);
void free_forward_buffers(CNN *net);
void free_cnn(CNN* net);

#endif

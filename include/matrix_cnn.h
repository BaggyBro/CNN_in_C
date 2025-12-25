#ifndef MATRIX_CNN_H
#define MATRIX_CNN_H

double** convolve_forward(double **input, int in_size, double **kernel, int k_size, int *out_size);
void convolve_backward(double **d_out, int out_size, double **input, int in_size,
                       double **kernel, int k_size, double **d_input, double **d_kernel);

void relu_forward(double **matrix, int rows, int cols);
void relu_backward(double **matrix, double **d_out, int rows, int cols);

double** maxpool_forward(double **input, int in_size, int pool_size, int *out_size, int ***mask);
void maxpool_backward(double **d_out, int out_size, int pool_size, int in_size, int **mask, double **d_input);

double* dense_forward(double *input,int input_size, double **weights, double *bias, int output_size);
void dense_backward(double *d_out, double *input, int input_size,
                    double **weights, int output_size,
                    double *d_input, double **d_weights, double *d_bias);

double* softmax_forward(double *input, int size);
double softmax_cross_entropy_loss_and_grad(double *pred_logits, int size, int target_label, double *d_logits_out);

#endif

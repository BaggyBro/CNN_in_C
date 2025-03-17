#ifndef MATRIX_CNN
#define MATRIX_CNN

double** convolve(double** input, int rows, int cols,  int in_size, double** kernel, int k_size);
void relu(double** matrix, int rows, int cols);
double** max_pooling(double** input, int rows, int cols,  int in_size, int pool_size);
double* dense_layer( double *input, int size,  int input_size, double** weights, double* bias, int output_size);
double* softmax(double* input, int size);

#endif

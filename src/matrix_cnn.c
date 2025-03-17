#include "../include/matrix_cnn.h"
#include "../include/matrix.h"
#include <math.h>
#include <stdlib.h>

double** convolve(double **input, int rows, int cols,  int in_size, double **kernel, int k_size){
  int out_size = in_size - k_size + 1;      // output matrix size
  double** output = allocate_matrix(rows, cols);
  
  for(int i = 0 ; i < out_size; i++){
    for(int j = 0 ; j < out_size; j++){
      double sum = 0.0;

      // Kernel applied here
      for(int ki = 0; ki < k_size; ki++){
        for(int kj = 0 ; kj < k_size; kj++){
          sum+= input[i+ki][j+kj] * kernel[ki][kj];
        }
      }

      output[i][j] = sum;
    }
  }

  return output;
}

void relu(double** matrix, int rows, int cols){
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      if(matrix[i][j] < 0) matrix[i][j] = 0;
    }
  }
}

double** max_pooling(double **input, int rows, int cols, int in_size, int pool_size){
  int out_size = in_size / pool_size;
  double** output = allocate_matrix(rows, cols);

  for(int i = 0 ; i < out_size; i++){
    for(int j = 0 ; j < out_size; j++){
      double max_value = input[i * pool_size][j * pool_size];

      for(int m = 0 ; m < pool_size; m++){
        for(int n = 0 ; n < pool_size; n++){
          double val = input[i * pool_size + m][j * pool_size +n];
          if(val > max_value) max_value = val;
        }  
      }

      output[i][j] = max_value;
    }

  }

  return output;
}

double* dense_layer(double *input,int size, int input_size, double **weights, double *bias, int output_size){

  double* output = (double*)malloc(size * sizeof(double));
  for(int i = 0; i < output_size; i++){
    output[i] = bias[i];

    for(int j = 0; j < input_size; j++){
      output[i] += input[j] * weights[i][j];
    }
  }

  return output;
}

double* softmax(double *input, int size){
  double* output = (double*)malloc(size * sizeof(double));
  double sum = 0.0;

  for(int i = 0 ; i < size; i++){
    output[i] = exp(input[i]);
    sum += output[i];
  }

  for(int i = 0 ; i < size; i++){
    output[i] /= sum;
  }

  return output;
}

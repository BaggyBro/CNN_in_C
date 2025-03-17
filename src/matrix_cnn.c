#include "../include/matrix_cnn.h"
#include "../include/matrix.h"

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

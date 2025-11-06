#include "../include/matrix_cnn.h"
#include "../include/matrix.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Convolution forward (single channel, stride=1, no padding)
   input: input[in_size][in_size]
   kernel: k_size x k_size
   returns output matrix and sets *out_size
*/
double** convolve_forward(double **input, int in_size, double **kernel, int k_size, int *out_size){
  int o = in_size - k_size + 1;
  *out_size = o;
  double** output = allocate_matrix(o, o);

  for(int i = 0; i < o; i++){
    for(int j = 0; j < o; j++){
      double sum = 0.0;
      for(int ki = 0; ki < k_size; ki++){
        for(int kj = 0; kj < k_size; kj++){
          sum += input[i+ki][j+kj] * kernel[ki][kj];
        }
      }
      output[i][j] = sum;
    }
  }
  return output;
}

/* Convolution backward: compute gradients w.r.t input and kernel
 d_out: gradient of loss w.r.t conv output of shape out_size x out_size
 input: original input (in_size x in_size)
 kernel: original kernel (k_size x k_size)
 d_input and d_kernel must be allocated by caller (sizes: in_size x in_size, k_size x k_size)
*/
void convolve_backward(double **d_out, int out_size, double **input, int in_size, double **kernel, int k_size, double **d_input, double **d_kernel){
  // zero d_input and d_kernel
  for(int i=0;i<in_size;i++) for(int j=0;j<in_size;j++) d_input[i][j] = 0.0;
  for(int i=0;i<k_size;i++) for(int j=0;j<k_size;j++) d_kernel[i][j] = 0.0;

  // gradient wrt kernel
  for(int i=0;i<out_size;i++){
    for(int j=0;j<out_size;j++){
      for(int ki=0; ki<k_size; ki++){
        for(int kj=0; kj<k_size; kj++){
          d_kernel[ki][kj] += d_out[i][j] * input[i+ki][j+kj];
        }
      }
    }
  }

  // gradient wrt input (full convolution of d_out with flipped kernel)
  for(int i=0;i<out_size;i++){
    for(int j=0;j<out_size;j++){
      for(int ki=0; ki<k_size; ki++){
        for(int kj=0; kj<k_size; kj++){
          d_input[i+ki][j+kj] += d_out[i][j] * kernel[ki][kj];
        }
      }
    }
  }
}

/* ReLU forward (in place)
*/
void relu_forward(double **matrix, int rows, int cols){
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      if(matrix[i][j] < 0.0) matrix[i][j] = 0.0;
    }
  }
}

/* ReLU backward (in-place use of original matrix values): d_out holds gradient coming from next layer.
   This modifies d_out to zero where input was <= 0.
   NOTE: If you change input in-place with relu_forward above, you should store a mask before relu; here we assume
   relu_forward was applied to the matrix so entries <=0 are zero. We'll assume "matrix" is the post-relu values;
   better practice: keep pre-activation saved. For simplicity, we use matrix post-activation to compute mask.
*/
void relu_backward(double **matrix, double **d_out, int rows, int cols){
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      if(matrix[i][j] <= 0.0) d_out[i][j] = 0.0;
    }
  }
}

/* Max-pooling forward (non-overlapping)
   returns output and mask to be used for backward (mask stores flattened index of winning element)
   mask is allocated inside (as int*[out_size]) and must be freed by caller:
     mask is a 2D int array of size out_size x out_size storing an integer pair index = m*pool_size + n (0..pool_size*pool_size-1)
*/
double** maxpool_forward(double **input, int in_size, int pool_size, int *out_size, int ***mask){
  int o = in_size / pool_size;
  *out_size = o;
  double** output = allocate_matrix(o, o);

  // allocate mask
  int **m = malloc(o * sizeof(int*));
  for(int i=0;i<o;i++){
    m[i] = malloc(o * sizeof(int));
  }

  for(int i=0;i<o;i++){
    for(int j=0;j<o;j++){
      double maxv = input[i*pool_size][j*pool_size];
      int max_idx = 0;
      for(int p=0;p<pool_size;p++){
        for(int q=0;q<pool_size;q++){
          double val = input[i*pool_size + p][j*pool_size + q];
          if(val > maxv){
            maxv = val;
            max_idx = p*pool_size + q;
          }
        }
      }
      output[i][j] = maxv;
      m[i][j] = max_idx;
    }
  }
  *mask = m;
  return output;
}

/* Maxpool backward: d_out is out_size x out_size, d_input is in_size x in_size and must be zeroed by caller
   mask is the array returned from forward.
*/
void maxpool_backward(double **d_out, int out_size, int pool_size, int in_size, int ***mask, double **d_input){
  // zero d_input
  for(int i=0;i<in_size;i++) for(int j=0;j<in_size;j++) d_input[i][j] = 0.0;

  int **m = *mask;
  for(int i=0;i<out_size;i++){
    for(int j=0;j<out_size;j++){
      int idx = m[i][j];
      int p = idx / pool_size;
      int q = idx % pool_size;
      int in_i = i*pool_size + p;
      int in_j = j*pool_size + q;
      d_input[in_i][in_j] += d_out[i][j];
    }
  }
}

/* Dense forward: input is length input_size, weights is output_size x input_size, bias length output_size
   returns allocated output length output_size
*/
double* dense_forward(double *input,int input_size, double **weights, double *bias, int output_size){
  double* output = (double*)malloc(output_size * sizeof(double));
  for(int i = 0; i < output_size; i++){
    output[i] = bias[i];
    for(int j = 0; j < input_size; j++){
      output[i] += input[j] * weights[i][j];
    }
  }
  return output;
}

/* Dense backward:
   d_out length = output_size
   input length = input_size
   weights is output_size x input_size
   returns d_input (allocated by caller, size input_size)
   computes d_weights (allocated by caller [output_size x input_size]) and d_bias [output_size]
*/
void dense_backward(double *d_out, double *input, int input_size, double **weights, int output_size, double *d_input, double **d_weights, double *d_bias){
  // d_input = W^T * d_out
  for(int i=0;i<input_size;i++) d_input[i] = 0.0;
  for(int i=0;i<output_size;i++){
    d_bias[i] = d_out[i];
    for(int j=0;j<input_size;j++){
      d_weights[i][j] = d_out[i] * input[j];
      d_input[j] += weights[i][j] * d_out[i];
    }
  }
}

/* Softmax forward (returns newly allocated array)
*/
double* softmax_forward(double *input, int size){
  double maxv = input[0];
  for(int i=1;i<size;i++) if(input[i] > maxv) maxv = input[i];
  double sum = 0.0;
  double *out = (double*)malloc(size * sizeof(double));
  for(int i=0;i<size;i++){
    out[i] = exp(input[i] - maxv);
    sum += out[i];
  }
  for(int i=0;i<size;i++) out[i] /= sum;
  return out;
}

/* Combined softmax + cross-entropy gradient:
   pred_logits: logits (not probabilities), size
   target_label: integer 0..size-1
   returns loss (scalar), writes gradient into d_logits_out (caller-allocated length size)
   d_logits_out = softmax(pred_logits) - one_hot(target)
*/
double softmax_cross_entropy_loss_and_grad(double *pred_logits, int size, int target_label, double *d_logits_out){
  double maxv = pred_logits[0];
  for(int i=1;i<size;i++) if(pred_logits[i] > maxv) maxv = pred_logits[i];

  double sum = 0.0;
  for(int i=0;i<size;i++){
    d_logits_out[i] = exp(pred_logits[i] - maxv);
    sum += d_logits_out[i];
  }
  for(int i=0;i<size;i++) d_logits_out[i] /= sum;

  double loss = -log(d_logits_out[target_label] + 1e-12);
  d_logits_out[target_label] -= 1.0; // now holds gradient: softmax - y
  return loss;
}

#include <stdio.h>
#include <stdlib.h>
#include "../include/matrix.h"

double** allocate_matrix(int rows, int cols){
  double** matrix = malloc(rows * sizeof(double*));
  for(int i = 0; i < rows; i++){
    matrix[i] = malloc(cols * sizeof(double));
  }
  return matrix;
}

void free_matrix(double **matrix, int rows){
  for(int i = 0; i < rows; i++){
    free(matrix[i]);
  }

  free(matrix);
}


double** matrix_add(double** a, double** b, int rows, int cols){
  double** result = allocate_matrix(rows, cols);
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      result[i][j] = a[i][j] + b[i][j];
    }
  }

  return result;
}

double** matrix_sub(double** a, double** b, int rows, int cols){
  double** result = allocate_matrix(rows, cols);
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      result[i][j] = a[i][j] - b[i][j];
    }
  }

  return result;
}

double** matrix_element_mult(double** a, double** b, int rows, int cols){
  double** result = allocate_matrix(rows, cols);
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      result[i][j] = a[i][j] * b[i][j];
    }
  }

  return result;
}

double** matrix_dot_mult(double** a, double** b, int rows_a, int cols_a, int rows_b, int cols_b){
  double** result = allocate_matrix(rows_a, cols_b);
  for(int i = 0 ; i < rows_a; i++){
    for(int j = 0 ; j < cols_b; j++){
      result[i][j] = 0;
      for(int k = 0 ; k < cols_a; k++){
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }

  return result;
}

double** transpose_matrix(double **matrix, int rows, int cols){
  double** result = allocate_matrix( cols, rows);
  for(int i = 0 ; i < cols; i++){
    for(int j = 0; j < rows; j++){
      result[i][j] = matrix[j][i];
    }
  }
 
  return result;
}


double* flatten_matrix(double **matrix, int rows, int cols) {
    double* result = (double*)malloc(rows * cols * sizeof(double));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i * cols + j] = matrix[i][j];
        }
    }

    return result;
}

double** matrix_reshape(double **matrix, int rows, int cols, int rows_n, int cols_n){
  double** result = allocate_matrix(rows_n, cols_n);
  int index = 0; 
  for(int i = 0; i < rows_n; i++){
    for(int j = 0; j < cols_n; j++){
      int oldi = index / cols;
      int oldj = index % cols;
      result[i][j] = matrix[oldi][oldj];
      index++; 
    }
  }
  return result;
}

void print_matrix(double **matrix, int rows, int cols){
  for(int i = 0 ; i < rows; i++ ){
    for(int j = 0; j < rows; j++){
      printf("%lf ", matrix[i][j]);
    }
    printf("\n");
  }
}

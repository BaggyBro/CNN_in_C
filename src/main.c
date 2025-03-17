#include "../include/matrix.h"
#include <stdio.h>
#include <stdlib.h>

int main(){
  double ** A = allocate_matrix(2, 2);
  double ** B = allocate_matrix(2, 2);

    A[0][0] = 1.0; A[0][1] = 2.0;
    A[1][0] = 3.0; A[1][1] = 4.0;

    B[0][0] = 5.0; B[0][1] = 6.0;
    B[1][0] = 7.0; B[1][1] = 8.0;

  double* C = flatten_matrix(A, 2, 2);
  double** D = transpose_matrix(A, 2, 2);    
  for(int i = 0 ;i < 4; i++){
    printf("%f ", C[i]);
  }
  print_matrix(D, 2, 2);

  free_matrix(A, 2);
  free_matrix(B, 2);
  free(C);
  free_matrix(D, 2);
  return 0;

}

#ifndef MATRIX_H
#define MATRIX_H

double** allocate_matrix(int rows, int cols);
void free_matrix(double** matrix, int rows);
double** matrix_add(double** a, double** b, int rows, int cols);
void print_matrix(double**matrix, int rows, int cols);
double** transpose_matrix(double** matrix, int rows, int cols);
double* flatten_matrix(double** matrix, int rows, int cols);
double** reshape_matric(double** matrix, int rows, int cols);

#endif

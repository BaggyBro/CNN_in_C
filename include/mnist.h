#ifndef MNIST_H
#define MNIST_H

typedef struct {
    int count;
    double ***images;
    int *labels;
} MNIST;

MNIST load_mnist(const char *image_path, const char* label_path);
void free_mnist(MNIST *d);

#endif

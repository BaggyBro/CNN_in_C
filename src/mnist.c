#include "../include/mnist.h"
#include "../include/matrix.h"
#include <stdio.h>
#include <stdlib.h>

static int read_int(FILE *f){
    unsigned char b[4];
    fread(b,1,4,f);
    return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3];
}

MNIST load_mnist(const char *image_path, const char *label_path){
    MNIST d;
    FILE *fi = fopen(image_path,"rb");
    FILE *fl = fopen(label_path,"rb");

    if(!fi || !fl){
        fprintf(stderr, "Error: Could not open MNIST files: %s or %s\n", image_path, label_path);
        d.count = 0;
        d.images = NULL;
        d.labels = NULL;
        return d;
    }

    read_int(fi);             // magic
    d.count = read_int(fi);  // num images
    read_int(fi); read_int(fi); // rows cols

    read_int(fl); read_int(fl); // magic + count

    d.images = malloc(d.count * sizeof(double**));
    d.labels = malloc(d.count * sizeof(int));

    for(int n=0;n<d.count;n++){
        d.images[n] = allocate_matrix(28,28);
        for(int i=0;i<28;i++){
            for(int j=0;j<28;j++){
                unsigned char p;
                fread(&p,1,1,fi);
                d.images[n][i][j] = p / 255.0;
            }
        }
        unsigned char lab;
        fread(&lab,1,1,fl);
        d.labels[n] = lab;
    }

    fclose(fi);
    fclose(fl);
    return d;
}

void free_mnist(MNIST *d){
    for(int i=0;i<d->count;i++)
        free_matrix(d->images[i],28);
    free(d->images);
    free(d->labels);
}

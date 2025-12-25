#include "../include/matrix.h"
#include "../include/cnn_model.h"
#include "../include/mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    srand(time(NULL));

    printf("Loading MNIST...\n");
    MNIST train = load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    MNIST test  = load_mnist("t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte");

    CNN *net = init_cnn();

    int epochs = 5;
    double lr = 0.01;

    for(int e=0;e<epochs;e++){
        double loss_sum = 0.0;

        for(int i=0;i<train.count;i++){
            double loss = forward_cnn(net, train.images[i], train.labels[i]);
            backward_cnn(net, train.images[i]);
            update_params(net, lr);
            free_forward_buffers(net);
            loss_sum += loss;

            if(i % 1000 == 0)
                printf("Epoch %d | sample %d | loss %.6f\n", e+1, i, loss);
        }

        printf("Epoch %d avg loss: %.6f\n", e+1, loss_sum / train.count);
    }

    int correct = 0;
    for(int i=0;i<test.count;i++){
        forward_cnn(net, test.images[i], test.labels[i]);
        int best = 0;
        for(int j=1;j<10;j++)
            if(net->softmax_out[j] > net->softmax_out[best])
                best = j;

        if(best == test.labels[i]) correct++;
        free_forward_buffers(net);
    }

    printf("Test Accuracy: %.2f%%\n", 100.0 * correct / test.count);

    free_cnn(net);
    free_mnist(&train);
    free_mnist(&test);

    return 0;
}

CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g -Iinclude
LDFLAGS = -lm

TARGET = cnn

SRC = src/main.c \
      src/matrix.c \
      src/matrix_cnn.c \
      src/cnn_model.c \
      src/mnist.c

OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean: 
	rm -f $(OBJ) $(TARGET)

run: all
	./$(TARGET)

.PHONY: all clean run

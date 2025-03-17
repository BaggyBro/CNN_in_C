CC = gcc
CGLAGS = -Wall -Wextra -std=c11 -g

TARGET = main

SRC = src/main.c src/matrix.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJ)

%.o:%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean: 
	rm -f $(OBJ) $(TARGET)

run: all
	./main

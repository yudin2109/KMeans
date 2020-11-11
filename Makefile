CC=g++

#CFLAGS= -std=c++17 -g3 -fsanitize=thread -pthread
CFLAGS= -std=c++17 -O3 -fopenmp

all:
	$(CC) $(CFLAGS) clustering.cpp -o clustering

clean:
	rm -rf *.o
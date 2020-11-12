CC = g++

CFLAGS = -std=c++17 -O2 -fopenmp

all:
	$(CC) $(CFLAGS) clustering.cpp -o clustering

clean:
	rm -rf *.o
CC = gcc
CFLAGS = -O2 -g
OPENMPFLAGS ?= -fopenmp
LDFLAGS ?=

main: main.o
	$(CC) -o main main.o $(LDFLAGS) $(OPENMPFLAGS) -lm
.c.o:
	$(CC) -c $< $(OPENMPFLAGS) $(CFLAGS)
clean:
	rm -f main main.o

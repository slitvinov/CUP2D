.POSIX:
.SUFFIXES:
.SUFFIXES: .c .o

CC = gcc
CFLAGS = -O2 -g
OPENMPFLAGS ?= -fopenmp

O = main.o

main: $O
	$(CC) -o main $O $(LDFLAGS) $(OPENMPFLAGS) -lm
.c.o:
	$(CC) -c $< $(OPENMPFLAGS) $(CFLAGS)
clean:
	rm -f main $O

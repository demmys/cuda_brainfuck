NVCC = nvcc
CFLAGS = -gencode=arch=compute_20,code=\"sm_20,compute_20\"

TARGET = gpubf
OBJS = print.o stopwatch.o transmit.o brainfuck.o main.o

.SUFFIXES: .cu .o
.PHONY: all run clean clear

all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $(TARGET) $(OBJS)

.cu.o:
	$(NVCC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS)

clear: clean
	rm -f $(TARGET)

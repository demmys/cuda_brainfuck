NVCC = nvcc

TARGET = gpuloop
OBJS = ../stopwatch.o loop.o 

.SUFFIXES: .cu .o
.PHONY: all run ptx clean clear

all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $(TARGET) $(OBJS)

.cu.o:
	$(NVCC) -c $<

ptx: loop.cu
	$(NVCC) -o loop.ptx -ptx $<

clean:
	rm -f $(OBJS)

clear: clean
	rm -f $(TARGET)

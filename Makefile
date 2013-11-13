NVCC = nvcc
CFLAGS = -gencode=arch=compute_20,code=\"sm_20,compute_20\" -rdc=true
LDFLAGS = -lcudart

TARGET = gpubf
OBJS = util.o transmit.o compile.o run.o brainfuck.o main.o

.SUFFIXES: .cu .o
.PHONY: all run clean clear

all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(OBJS)

.cu.o:
	$(NVCC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS)

clear: clean
	rm -f $(TARGET)

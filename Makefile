NVCC = nvcc
FLAGS = -gencode=arch=compute_20,code=\"sm_20,compute_20\"

TARGET = gpubf
OBJS = main.o brainfuck.o transmit.o util.o

.SUFFIXES: .cu .o
.PHONY: all run clean clear

all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $(TARGET) $(OBJS)

.cu.o:
	$(NVCC) $(FLAGS) -c $<

clean:
	rm -f $(OBJS)

clear: clean
	rm -f $(TARGET)

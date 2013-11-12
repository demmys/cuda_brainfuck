CC = nvcc

TARGET = gpubf
SOURCE = brainfuck.cu transmit.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) -o $@ $^

clean:
	rm -f $(TARGET)

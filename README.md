Brainfuck interpreter on GPGPU with CUDA
-----

##System Requirements
NVIDIA GPU CUDA Compute Capability >=2.0

##Usage
USAGE: gpubf [-chmntv] [-d character] [file ...]

OPTIONS:
    -c      Execute the same program on CPU(1 core, 1 thread).
    -h      Display available options and exit.
    -l      Display execution time and number of processor for logging.
    -m      Display execution time includes copying memory among host and device.
            This option turns on the -t option.
    -n      Display the result of execution with a number.
    -t      Display execution time with the result.
    -v      Display product version and exit.
    -b decimal
            Set the block size of kernel function
            (default block size is 1).
    -d character
            Set the delimiter of the source code
            (default delimiter is LF, and EOF is always taken as the delimiter).

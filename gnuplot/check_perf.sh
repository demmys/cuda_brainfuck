#!/bin/bash

for i in `seq 1 10`
do
    cp 1000000.bf tester.bf
    vim -S copy_double.vim tester.bf
    perf stat ./gpubf -nl tester.bf
done

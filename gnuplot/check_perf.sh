#!/bin/bash

cp 1000000.bf tester.bf
vim -c 'normal ggyy15p' -c 'wq' tester.bf
touch perf_result.log

for i in `seq 1 64`
do
    perf stat ./gpubf -cntl tester.bf >> perf_result.log 2>&1
    vim -c 'normal ggyy16p' -c 'wq' tester.bf
done

rm tester.bf
less perf_result.log

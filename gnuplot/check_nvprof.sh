#!/bin/bash

bf=tester.bf
cp 1000000.bf $bf
vim -c 'normal ggyy15p' -c 'wq' $bf
res=nvprof_result.log
touch $res

for i in `seq 1 64`
do
    nvprof --print-gpu-trace ./gpubf -tl $bf >> $res 2>&1
    vim -c 'normal ggyy16p' -c 'wq' $bf
done

rm $bf

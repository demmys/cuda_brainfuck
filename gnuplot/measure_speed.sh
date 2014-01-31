#!/bin/bash

echo "processor $processor start."
dir=${processor}_speed_tick1
mkdir $dir
average="${dir}/average_tick1.log"
touch $average

bf=tester.bf
cp 500000.bf $bf

for i in `seq 1 256`
do
    thread=$i
    file="${dir}/${thread}thread.log"
    touch $file
    echo "processing ${file}..."
    for j in `seq 1 10`
    do
        echo -n "${j}"
        if test "$processor" = "cpu"
        then
            ./gpubf -ctl $bf >> $file
        else
            ./gpubf -tl $bf >> $file
        fi
        if test $j = 10
        then
            echo
        else
            echo -n ", "
        fi
    done
    cat $file
    node make_average.js $file >> $average
    vim -c 'normal ggyyp' -c 'wq' $bf
done

rm $bf
echo "processor $processor end."

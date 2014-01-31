#!/bin/bash

command="./gpuloop"

echo "processor $processor with $optimize start."
dir=${processor}_loop_speed_${optimize}
mkdir $dir
average="${dir}/average.log"
touch $average

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
            if test "$optimize" = "optimized"
            then
                $command -clo -t $i >> $file
            else
                $command -cl -t $i >> $file
            fi
        else
            if test "$optimize" = "optimized"
            then
                $command -lo -t $i >> $file
            else
                $command -l -t $i >> $file
            fi
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
done

echo "processor $processor end."

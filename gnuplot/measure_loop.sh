#!/bin/bash

processor=cpu optimize=not-optimized ./measure_loop_speed.sh
processor=cpu optimize=optimized ./measure_loop_speed.sh

processor=gpu optimize=not-optimized ./measure_loop_speed.sh
processor=gpu optimize=optimized ./measure_loop_speed.sh

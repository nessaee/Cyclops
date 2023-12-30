#!/bin/bash

# Execute Python commands
python ./main.py --ab -k 10 -K 6  -s 1 -r 1 -t 1 --save "sim-1" --process
python ./main.py --ab -k 15 -K 4  -s 1 -r 1 -t 1 --save "sim-1" --process
python ./main.py --ab -k 6 -K 10  -s 1 -r 1 -t 1 --save "sim-1" --process
python ./main.py --ab -k 4 -K 15  -s 1 -r 1 -t 1 --save "sim-1" --process


python ./main.py --ab -k 10 -K 6  -s 1 -r 2 -t 0.5 --save "sim-2" --process
python ./main.py --ab -k 15 -K 4  -s 1 -r 2 -t 0.5 --save "sim-2" --process
python ./main.py --ab -k 6 -K 10  -s 1 -r 2 -t 0.5 --save "sim-2" --process
python ./main.py --ab -k 4 -K 15  -s 1 -r 2 -t 0.5 --save "sim-2" --process

python ./main.py --ab -k 10 -K 6  -s 1 -r 4 -t 0.25 --save "sim-4" --process
python ./main.py --ab -k 15 -K 4  -s 1 -r 4 -t 0.25 --save "sim-4" --process
python ./main.py --ab -k 6 -K 10  -s 1 -r 4 -t 0.25 --save "sim-4" --process
python ./main.py --ab -k 4 -K 15  -s 1 -r 4 -t 0.25 --save "sim-4" --process

python ./main.py --ab -k 10 -K 6  -s 1 -r 8 -t 0.125 --save "sim-8" --process
python ./main.py --ab -k 15 -K 4  -s 1 -r 8 -t 0.125 --save "sim-8" --process
python ./main.py --ab -k 6 -K 10  -s 1 -r 8 -t 0.125 --save "sim-8" --process
python ./main.py --ab -k 4 -K 15  -s 1 -r 8 -t 0.125 --save "sim-8" --process

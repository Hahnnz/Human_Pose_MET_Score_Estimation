#!/bin/sh

gpuNum = $1

python3 ./scripts/train/Human_Joint_Pointor.py $1 128,128 800 10000 0.7 100 0.01

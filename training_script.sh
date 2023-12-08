#!/bin/zsh

python train.py 'US-101' 5000 3500 'LSTMHMM'

python train.py 'US-101' 5000 3500 'LSTM'

python train.py 'US-101' 5000 3500 'HMM'

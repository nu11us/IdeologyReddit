#!/bin/bash

killall tensorboard
tensorboard --logdir=./runs --bind_all &
python pol_model.py

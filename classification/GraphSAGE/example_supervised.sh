#!/usr/bin/env bash

echo '' > ./stats.txt
for i in `seq 0 9`
do
python -m graphsage.supervised_train --train_prefix ./data/users_anon --model graphsage_mean --epochs 6 --w1 36 --batch_size 256 --dropout 0.1 --fold $i
done
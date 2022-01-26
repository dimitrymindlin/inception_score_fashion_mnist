#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate inception-score-fashion-mnist

cd /media/compute/homes/dmindlin/inception_score_fashion_mnist

tensorboard dev upload --logdir logs \
    --name "Inception FashionMnist" \
    --description "Trained Inception on FashionMnist." \
    --one_shot

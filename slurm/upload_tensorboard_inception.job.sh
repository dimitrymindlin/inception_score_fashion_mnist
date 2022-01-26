#!/bin/bash
source /media/compute/homes/dmindlin/.bashrc
conda activate medical-decision-support-system

cd /media/compute/homes/dmindlin/medical-decision-support-system

tensorboard dev upload --logdir logs \
    --name "Inception FashionMnist" \
    --description "Trained Inception on FashionMnist." \
    --one_shot

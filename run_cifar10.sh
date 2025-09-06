no #!/bin/bash

export PYTHONPATH=$PYTHONPATH:..

python -m pruning_identified_exemplars.save_checkpoint.train_eval \
  --dataset=cifar10 \
  --model_name=wide_resnet \
  --widenet_depth=28 \
  --widenet_width=10 \
  --mode=train \
  --output_dir=/tmp/cifar10_train_eval \
  --test_small_sample=True

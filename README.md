pip install torch torchvision pytorch-lightning pycocotools


Training & Resume:

python train.py --data_dir /path/to/coco/train2017 --ann_file /path/to/instances_train2017.json --max_epochs 10


To resume training from a checkpoint, add:

--resume_checkpoint /path/to/checkpoint.ckpt

Hyperparameter Tuning:

To run an Optuna study:

python train.py --data_dir /path/to/coco/train2017 --ann_file /path/to/instances_train2017.json --tune


https://bbycroft.net/llm



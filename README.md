# AutoDeiT

**A NAS method for searching DeiT**

AutoDeiT is a method for searching the optimal DeiT architecture and its corresponding optimal knowledge distillation teacher model under parameter constraints.

<div align="center">
    <img width="50%" alt="AutoDeiT" src="https://github.com/NASdn6hyc8/AutoDeiT/releases/download/model/SuperDeiT.png">
</div>

## Our Trained Model/Checkpoint

### SuperDeiT

Our trained SuperDeiT weight is available in [GitHub](https://github.com/NASdn6hyc8/AutoDeiT/releases/download/model/superDeiT.pth), which can be used for searching and finetuning.

### Search Result

Our search result is available in [GitHub](https://github.com/NASdn6hyc8/AutoDeiT/releases/download/model/checkpoint-20.pth.tar), which contains the information of the final generation population from evolutionary search.

### SubDeiT

Our finetuned SubDeiT weight is available in [GitHub](https://github.com/NASdn6hyc8/AutoDeiT/releases/download/model/subDeiT.pth).

## Usage

### Data Preparation

You need to download the [Imagenet-1k](http://www.image-net.org/) and move images to labeled subfolders.

Here is a example of directory structure:

```
/PATH/TO/IMAGENET/
  train/
    class_1/
      img_1.jpeg
    class_2/
      img_2.jpeg
    ...
  val/
    class_1/
      img_3.jpeg
    class_2/
      img_4.jpeg
    ...
```

### Test

To test our SubDeiT, you need to move the downloaded model to `/PATH/TO/CHECKPOINT`.

Then you can test our SubDeiT with following command:
```bulidoutcfg
python -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path /PATH/TO/IMAGENET/ --gp --change_qkv --mode retrain --relative_position --dist-eval --cfg ./experiments/subnet/AutoDeiT-T.yaml --resume /PATH/TO/CHECKPOINT --eval --teacher_model ''


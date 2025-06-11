# AutoDeiT

**A NAS method for searching DeiT**

AutoDeiT is a method for searching the optimal DeiT architecture and its corresponding optimal knowledge distillation teacher model under parameter constraints.

## Our Trained Model/Checkpoint

### SuperDeiT

Our trained SuperDeiT weight will be available in [GitHub](https://github.com/NASdn6hyc8/AutoDeiT/releases/download/model/superDeiT.pth), which can be used for searching and finetuning.

### Search Result

Our search result will be available in [GitHub](https://github.com/NASdn6hyc8/AutoDeiT/releases/download/model/checkpoint-20.pth.tar), which contains the information of the final generation population from evolutionary search.

### SubDeiT

Our finetuned SubDeiT weight will be available in [GitHub](https://github.com/NASdn6hyc8/AutoDeiT/releases/download/model/subDeiT.pth).

## Usage

### Data Preparation

You need to download the [Imagenet-1k](http://www.image-net.org/) and move images to labeled subfolders.

Here is a example of directory structure:

```
/PATH/TO/IMAGENET/
  train/
    class_1/
      img_1.jpeg
      img_2.jpeg
      ...
    class_2/
      img_3.jpeg
      ...
    ...
  val/
    class_1/
      img_4.jpeg
      img_5.jpeg
      ...
    class_2/
      img_6.jpeg
      ...
    ...
```
### Supernet Training
You can train/fine-tune a supernet with following command:

```bulidoutcfg

python -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path /PATH/TO/IMAGENET/ --gp --change_qkv --mode retrain --relative_position --dist-eval --cfg ./experiments/subnet/AutoDeiT-T.yaml --resume /PATH/TO/CHECKPOINT --eval

```

### Test

To test our SubDeiT, you need to move the downloaded SubDeiT model to `/PATH/TO/CHECKPOINT`.

Then you can test our SubDeiT with following command:
```bulidoutcfg

python -m torch.distributed.launch --nproc_per_node=4 --use_env supernet_train.py --data-path /PATH/TO/IMAGENET/ --gp --change_qkv --mode retrain --relative_position --dist-eval --cfg ./experiments/subnet/AutoDeiT-T.yaml --resume /PATH/TO/CHECKPOINT --eval

```

## Acknowledgements

The codes are inspired by [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer), [HAT](https://github.com/mit-han-lab/hardware-aware-transformers), [DeiT](https://github.com/facebookresearch/deit), [SPOS](https://github.com/megvii-model/SinglePathOneShot).

# Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer

This repository contains the PyTorch code for Evo-ViT (accepted by AAAI-22). 

This work proposes a slow-fast token evolution approach to accelerate vanilla vision transformers of both flat and deep-narrow structures without additional pre-training and fine-tuning procedures. For details please see [Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer](https://arxiv.org/abs/2108.01390) by Yifan Xu*, Zhijie Zhang*, Mengdan Zhang, Kekai Sheng, Ke Li, Weiming Dong, Liqing Zhang, Changsheng Xu, and Xing Sun.
![intro](imgs/method.png)

Our code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), and [LeViT](https://github.com/facebookresearch/LeViT).

# Preparation
Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is the standard layout for the torchvision datasets.ImageFolder, and the training and validation data is expected to be in the train/ folder and val folder respectively.
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
All distillation settings are conducted with a teacher model RegNetY-160, which is available at [teacher checkpoint](https://dl.fbaipublicfiles.com/deit/regnety_160-a5fe301d.pth).

Install the requirements by running:
```
pip3 install -r requirements.txt
```
NOTE that all experiments in the paper are conducted under cuda11.0. If necessary, please install the following packages under the environment with CUDA version 11.0:
[torch1.7.0-cu110](https://download.pytorch.org/whl/cu110/torch-1.7.0%2Bcu110-cp36-cp36m-linux_x86_64.whl), 
[torchvision-0.8.1-cu110](https://download.pytorch.org/whl/cu110/torchvision-0.8.1%2Bcu110-cp36-cp36m-linux_x86_64.whl).

# Model Zoo

We provide our Evo-ViT models pretrained on ImageNet:
| Name            | Top-1 Acc (\%) | Throughput (img/s)   | Url                                                                                                |
| --------------- | -------------- | -------------------- | -------------------------------------------------------------------------------------------------- |
| Evo-ViT-T       |  72.0          |     4027             | [Google Drive](https://drive.google.com/file/d/1AL4uHGHvCoFXkrtHRgf4XmurgRs5-qab/view?usp=sharing) |
| Evo-ViT-S       |  79.4          |     1510             | [Google Drive](https://drive.google.com/file/d/1AiD1J-z9klr72-zczkJzX1HHVlxO7iin/view?usp=sharing) |
| Evo-ViT-B       |  81.3          |     462              | [Google Drive](https://drive.google.com/file/d/15EmMKb4L5IjHqnMYQHVNYZGRRGdByTsz/view?usp=sharing) |
| Evo-LeViT-128S  |  73.0          |     10135            | [Google Drive](https://drive.google.com/file/d/1urqO1OqpMK8_Y3E7hQilLkCT9hsTv7ST/view?usp=sharing) |
| Evo-LeViT-128   |  74.4          |     8323             | [Google Drive](https://drive.google.com/file/d/1rvMe1Iz_9d6meAbbov25pblCeQ7Q0tt2/view?usp=sharing) |
| Evo-LeViT-192   |  76.8          |     6148             | [Google Drive](https://drive.google.com/file/d/1tWcgm3Z3WSaY4awycLwY4Jc1j_dIIOxn/view?usp=sharing) |
| Evo-LeViT-256   |  78.8          |     4277             | [Google Drive](https://drive.google.com/file/d/1CG-MLsPhKzs1CI613sQoW0Q7fPUuLBOP/view?usp=sharing) |
| Evo-LeViT-384   |  80.7          |     2412             | [Google Drive](https://drive.google.com/file/d/1cFmHWSCHeTaS4o5zL_qAc2tkWUugxvQh/view?usp=sharing) |
| Evo-ViT-B*      |  82.0          |     139              | [Google Drive](https://drive.google.com/file/d/1MBSH4Fx8Bq9cgGhktvYAb23EFioiv2y2/view?usp=sharing) |
| Evo-LeViT-256*  |  81.1          |     1285             | [Google Drive](https://drive.google.com/file/d/1MHHljQCzz-L6Sj18vq7QrQHhn4YvWjn8/view?usp=sharing) |
| Evo-LeViT-384*  |  82.2          |     712              | [Google Drive](https://drive.google.com/file/d/1YzLPOLMLSymLzIdMqYol1HI7MwO35_7j/view?usp=sharing) |

The input image resolution is 224 × 224 unless specified. \* denotes the input image resolution is 384 × 384. 

# Usage

## Evaluation
To evaluate a pre-trained model, run:
```
python3 main_deit.py --model evo_deit_small_patch16_224 --eval --resume /path/to/checkpoint.pth --batch-size 256 --data-path /path/to/imagenet
```

## Training with input resolution of 224
To train Evo-ViT  on ImageNet on a single node with 8 gpus for 300 epochs,  run:
 
Evo-ViT-T
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_deit.py --model evo_deit_tiny_patch16_224 --drop-path 0 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```

Evo-ViT-S
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_deit.py --model evo_deit_small_patch16_224 --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/save
```

Sometimes loss Nan happens in the early training epochs of DeiT-B, which is described in this [issue](https://github.com/facebookresearch/deit/issues/29). Our solution is to reduce the batch size to 128, load a [warmup checkpoint](https://drive.google.com/file/d/1k3luEHWyQ7HuU6g1pmh2f2gDDOPqQmb5/view?usp=sharing) trained for 9 epochs, and train Evo-ViT for the remaining 291 epochs. To train Evo-ViT-B  on ImageNet on a single node with 8 gpus for 300 epochs,  run:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_deit.py --model evo_deit_base_patch16_224 --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/save --resume /path/to/warmup_checkpoint.pth
```

To train Evo-LeViT-128  on ImageNet on a single node with 8 gpus for 300 epochs,  run:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_levit.py --model EvoLeViT_128 --batch-size 256 --data-path /path/to/imagenet --output_dir /path/to/save
```
The other models of Evo-LeViT are trained with the same command as mentioned above.

## Training with input  resolution of 384

To train Evo-ViT-B*  on ImageNet on 2 nodes with 8 gpus each for 300 epochs, run:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=$NODE_SIZE  --node_rank=$NODE_RANK --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR main_deit.py --model evo_deit_base_patch16_384 --input-size 384 --batch-size 64 --data-path /path/to/imagenet --output_dir /path/to/save
```

To train Evo-ViT-S*  on ImageNet on a single node with 8 gpus for 300 epochs,  run:
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_deit.py --model evo_deit_small_patch16_384 --batch-size 128 --input-size 384 --data-path /path/to/imagenet --output_dir /path/to/save"
```

To train Evo-LeViT-384*  on ImageNet on a single node with 8 gpus for 300 epochs,  run:

```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_levit.py --model EvoLeViT_384_384 --input-size 384 --batch-size 128 --data-path /path/to/imagenet --output_dir /path/to/save
```

The other models of Evo-LeViT* are trained with the same command of Evo-LeViT-384*.

## Testing inference throughput
To test inference throughput, first modify the model name in line 153 of benchmark.py. Then, run:
```
python3 benchmark.py
```
The defauld input resolution is 224. To test inference throughput with input resolution of 384, please add the parameter "--img_size 384"

## Visualization of token selection
The visualization code is modified from [DynamicViT](https://github.com/raoyongming/DynamicViT).

To visualize a batch of ImageNet val images, run:
```
python3 visualize.py --model evo_deit_small_vis_patch16_224 --resume /path/to/checkpoint.pth --output_dir /path/to/save --data-path /path/to/imagenet --batch-size 64 
```
To visualize a single image, run:
```
python3 visualize.py --model evo_deit_small_vis_patch16_224 --resume /path/to/checkpoint.pth --output_dir /path/to/save --img-path ./imgs/a.jpg --save-name evo_test
```
Add parameter '--layer-wise-prune' if the visualized model is not trained with layer-to-stage training strategy.

The visualization results of Evo-ViT-S are as follows:

![result](imgs/results.png)


# Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{evo-vit,
  title={Evo-vit: Slow-fast token evolution for dynamic vision transformer},
  author={Xu, Yifan and Zhang, Zhijie and Zhang, Mengdan and Sheng, Kekai and Li, Ke and Dong, Weiming and Zhang, Liqing and Xu, Changsheng and Sun, Xing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={3},
  pages={2964--2972},
  year={2022}
}
```

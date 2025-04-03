# (TGRS 2024) OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images

The Chinese Version is below (中文版在下面).

## 🌱 환경설정 (김예찬)

~~~shell
conda create --name orientedformer python=3.8 -y
conda activate orientedformer
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/torch_stable.html  
pip install -U openmim==0.3.9
mim install mmcv-full==1.7.2
pip install -v -e mmdetection/
pip install -v -e mmrotate/
pip install setuptools==59.5.0
~~~



## Introduction

The paper is officially accepted by IEEE Transactions on Geoscience and Remote Sensing (**TGRS 2024**).

TGRS paper link https://ieeexplore.ieee.org/document/10669376

arxiv link https://arxiv.org/abs/2409.19648

NEW

- [x] ICDAR2015 Dataset in MMRotate-1.x

- [x] ICDAR2015 Metric  in MMRotate-1.x

- [x] ChannelMapperWithGN in MMRotate-1.x

- [x] RBBoxL1Cost in MMRotate-1.x

- [x] RotatedIoUCost in  MMRotate-1.x

- [x] TopkHungarianAssigner in MMRotate-1.x

If you like it, please click on star.



## Installation

Please refer to [Installation](https://mmrotate.readthedocs.io/en/1.x/get_started.html) for more detailed instruction.

**Note**: Our codes base on the newest version mmrotate-1.x, not mmrotate-0.x.

**Note**: All of our codes can be found in [path](https://github.com/wokaikaixinxin/OrientedFormer/tree/main/projects/OrientedFormer) './projects/OrientedFormer/'.



## Data Preparation

DOTA and DIOR-R : Please refer to [Preparation](https://github.com/open-mmlab/mmrotate/tree/1.x/tools/data) for more detailed data preparation.

ICDAR2015 : (1) Download ICDAR2015 dataset from [official link](https://rrc.cvc.uab.es/?ch=4&com=introduction). (2) The data structure is as follows:

```bash
root
├── icdar2015
│   ├── ic15_textdet_train_img
│   ├── ic15_textdet_train_gt
│   ├── ic15_textdet_test_img
│   ├── ic15_textdet_test_gt
```



## Train

**1). DIOR-R**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py 2
```

**2). DOTA-v1.0**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_train.sh  projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

**3). DOTA-v1.5**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py 2
```

**4). DOTA-v2.0**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py 2
```

**5). ICDAR2015**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py 2
```



## Test

**1). DIOR-R**

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py work_dirs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py work_dirs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior/epoch_12.pth 2
```

**2). DOTA-v1.0**

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py work_dirs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py work_dirs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0/epoch_12.pth 2
```

Upload results to DOTA official [website](https://captain-whu.github.io/DOTA/evaluation.html).

**3). DOTA-v1.5**

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5/epoch_12.pth 2
```

Upload results to DOTA official [website](https://captain-whu.github.io/DOTA/evaluation.html).

**4). DOTA-v2.0**

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0/epoch_12.pth 2
```

Upload results to DOTA official [website](https://captain-whu.github.io/DOTA/evaluation.html).

**5). ICDAR2015**

Get result submit.zip

```
python tools/test.py projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015/epoch_21.pth
```

 Calculate precision, recall and F-measure. The script.py adapted from [official website](https://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1).

```
pip install Polygon3
python projects/icdar2015_evaluation/script.py –g=gt.zip –s=submit.zip
```



## Main Result

**1). DOTA-v1.0**

| Dataset   | **Configs**                                                  | Download                                                     | AP_50        | AP75      | mAP         | Backbone | lr schd | bs               |
| --------- | -------------------- | ------------------------------------------------------------ | ------------ | --------- | ----------- | -------- | ------- | ---------------- |
| DOTA-v1.0 | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0) | 75.3729      | 46.390216 | 45.0071     | R50      | 12epoch | 2img*2 rtx2080ti |
| DOTA-v1.0 | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms) | 79.064371    | 57.463    | 51.891899   | R50      | 12epoch | 2img*2 rtx2080ti |
| DOTA-v1.0 | [orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0) | 75.915958978 | 49.76108  | 47.11829758 | R101     | 12epoch | 2img*2 rtx2080ti |
| DOTA-v1.0 | [orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0) | 75.8819      | 48.965    | 45.8218     | Swin-T   | 12epoch | 2img*2 rtx2080ti |

**2). DOTA-v1.5**

| Dataset   | **Configs**                                                  | Download                                                     | AP_50 | Backbone | lr schd | bs               |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- | -------- | ------- | ---------------- |
| DOTA-v1.5 | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5) | 67.06 | R50      | 12epoch | 2img*2 rtx2080ti |

Due to the limitation of the length of the paper, all categories of AP for DOTA-1.5 are not available in the paper. Here is a list:

| PL      | BD      | BR      | GTF     | SV      | LV      | SH      | TC       | BC       | ST        | SBF       | RA     | HA        | SP        | HC      | CC        | AP50     | AP75    | mAP      |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | --------- | --------- | ------ | --------- | --------- | ------- | --------- | -------- | ------- | -------- |
| 72.0444 | 77.4554 | 51.2471 | 64.9538 | 64.0453 | 77.0387 | 85.3310 | 90.83699 | 77.31017 | 78.106886 | 56.103059 | 68.776 | 68.140988 | 72.081567 | 58.6135 | 10.855397 | 67.05879 | 39.2845 | 38.78675 |

**3). DOTA-v2.0**

| Dataset   | **Configs**                                                  | Download                                                     | AP_50 | Backbone | lr schd | bs               |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- | -------- | ------- | ---------------- |
| DOTA-v2.0 | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0) | 54.27 | R50      | 12epoch | 2img*2 rtx2080ti |

Due to the limitation of the length of the paper, all categories of AP for DOTA-2.0 are not available in the paper. Here is a list:

| PL      | BD       | BR         | GTF       | SV        | LV       | SH        | TC      | BC      | ST      | SBF     | RA        | HA      | SP       | HC       | CC     | airport  | helipad | AP50      | AP75       | mAP        |
| ------- | -------- | ---------- | --------- | --------- | -------- | --------- | ------- | ------- | ------- | ------- | --------- | ------- | -------- | -------- | ------ | -------- | ------- | --------- | ---------- | ---------- |
| 76.7619 | 51.55655 | 42.3872759 | 60.464159 | 56.482355 | 55.43076 | 66.681058 | 78.6341 | 60.0626 | 69.6894 | 35.0316 | 56.015956 | 51.9962 | 56.20235 | 54.95597 | 24.335 | 67.31572 | 12.9641 | 54.266644 | 28.8561385 | 30.0281367 |

**4). DIOR-R**

| Dataset | **Configs**                                                  | Download                                                     | AP_50 | Backbone | lr schd | bs               |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- | -------- | ------- | ---------------- |
| DIOR-R  | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior) | 67.28 | R50      | 12epoch | 2img*2 rtx2080ti |
| DIOR-R  | [orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior) | 68.84 | Swin-T   | 12epoch | 2img*2 rtx2080ti |
| DIOR-R  | [orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior) | 65.07 | LSK-Net  | 12epoch | 2img*2 rtx2080ti |

**5). ICDAR-2015**

| Dataset   | **Configs**                                                  | Download                                                     | P    | R    | F-measure | Backbone | lr schd | bs               |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- | ---- | --------- | -------- | ------- | ---------------- |
| ICDAR2015 | [orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015) | 85.3 | 74.2 | 79.4      | R50      | 24epoch | 2img*2 rtx2080ti |



## Cite OrientedFormer

“Instead of citing an article with closed or falsely advertised open-source code, it is preferable to cite an article with genuine open-source code.”

```
@ARTICLE{10669376,
  author={Zhao, Jiaqi and Ding, Zeyu and Zhou, Yong and Zhu, Hancheng and Du, Wen-Liang and Yao, Rui and El Saddik, Abdulmotaleb},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Encoding;Object detection;Proposals;Detectors;Remote sensing;Current transformers;Position measurement;End-to-end detectors;oriented object detection;positional encoding (PE);remote sensing;transformer},
  doi={10.1109/TGRS.2024.3456240}}
```

***



# (TGRS 2024) OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images



## 简介

论文被IEEE Transactions on Geoscience and Remote Sensing (**TGRS 2024**) 接受。

TGRS官方论文链接 https://ieeexplore.ieee.org/document/10669376

arxiv link https://arxiv.org/abs/2409.19648

新特性：

- [x] 数据集工具：                                 ICDAR2015 Dataset in MMRotate-1.x

- [x] 数据集工具：                                 ICDAR2015 Metric  in MMRotate-1.x

- [x] 不用FPN，用channelmapper： ChannelMapperWithGN in MMRotate-1.x

- [x] 端到端分配的L1代价矩阵：          RBBoxL1Cost in MMRotate-1.x

- [x] 端到端分配的IoU代价矩阵：        RotatedIoUCost in  MMRotate-1.x

- [x] Topk匈牙利匹配：                        TopkHungarianAssigner in MMRotate-1.x

如果喜欢，请点一点小星星收藏。

**“与其引用不开源代码、假开源代码的文章，不如引用真开源代码的文章。”**



## 安装

参考mmrotate-1.x的官方[安装教程](https://mmrotate.readthedocs.io/en/1.x/get_started.html)获取更多安装细节。

注意：代码是基于最新版本的mmrotate-1.x，而不是旧版的mmrotate-0.x。

注意：orientedformer的代码全部在[路径](https://github.com/wokaikaixinxin/OrientedFormer/tree/main/projects/OrientedFormer)'./projects/OrientedFormer/'



## 数据准备

DOTA and DIOR-R : 请参考mmrotate-1.x官方[数据处理](https://github.com/open-mmlab/mmrotate/tree/1.x/tools/data)对DOTA和DIOR-R的准备方法。

ICDAR2015 : (1) 从官方网站下载 [ICDAR2015](https://rrc.cvc.uab.es/?ch=4&com=introduction) 数据集。(2) 数据路径结构如下所示：

```bash
root
├── icdar2015
│   ├── ic15_textdet_train_img
│   ├── ic15_textdet_train_gt
│   ├── ic15_textdet_test_img
│   ├── ic15_textdet_test_gt
```



## 训练

**1). DIOR-R**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py 2
```

**2). DOTA-v1.0**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py 2
```

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

```bash
bash tools/dist_train.sh  projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py 2
```

**3). DOTA-v1.5**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py 2
```

**4). DOTA-v2.0**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py 2
```

**5). ICDAR2015**

```bash
bash tools/dist_train.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py 2
```



## 测试

**1). DIOR-R**

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py work_dirs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py work_dirs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior/epoch_12.pth 2
```

**2). DOTA-v1.0**

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py work_dirs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0/epoch_12.pth 2
```

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py work_dirs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0/epoch_12.pth 2
```

将结果上传 [DOTA](https://captain-whu.github.io/DOTA/evaluation.html)官方网站。

**3). DOTA-v1.5**

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5/epoch_12.pth 2
```

将结果上传 [DOTA](https://captain-whu.github.io/DOTA/evaluation.html)官方网站。

**4). DOTA-v2.0**

```bash
bash tools/dist_test.sh projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0/epoch_12.pth 2
```

将结果上传 [DOTA](https://captain-whu.github.io/DOTA/evaluation.html)官方网站。

**5). ICDAR2015**

得到结果submit.zip

```
python tools/test.py projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py work_dirs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015/epoch_21.pth
```

计算precision, recall 和 F-measure

```
pip install Polygon3
python projects/icdar2015_evaluation/script.py –g=gt.zip –s=submit.zip
```



## 主要结果

**1). DOTA-v1.0**

| Dataset   | **Configs**                                                  | Download                                                     | AP_50        | AP75      | mAP         | Backbone | lr schd | bs               |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------ | --------- | ----------- | -------- | ------- | ---------------- |
| DOTA-v1.0 | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0) | 75.3729      | 46.390216 | 45.0071     | R50      | 12epoch | 2img*2 rtx2080ti |
| DOTA-v1.0 | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.0-ms) | 79.064371    | 57.463    | 51.891899   | R50      | 12epoch | 2img*2 rtx2080ti |
| DOTA-v1.0 | [orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r101_q300_layer2_head64_point32_1x_dotav1.0) | 75.915958978 | 49.76108  | 47.11829758 | R101     | 12epoch | 2img*2 rtx2080ti |
| DOTA-v1.0 | [orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dotav1.0) | 75.8819      | 48.965    | 45.8218     | Swin-T   | 12epoch | 2img*2 rtx2080ti |

**2). DOTA-v1.5**

| Dataset   | **Configs**                                                  | Download                                                     | AP_50 | Backbone | lr schd | bs               |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- | -------- | ------- | ---------------- |
| DOTA-v1.5 | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav1.5) | 67.06 | R50      | 12epoch | 2img*2 rtx2080ti |

由于论文长度的限制，论文中没有DOTA-1.5的所有类别的AP，这里列出：

| PL      | BD      | BR      | GTF     | SV      | LV      | SH      | TC       | BC       | ST        | SBF       | RA     | HA        | SP        | HC      | CC        | AP50     | AP75    | mAP      |
| ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- | -------- | --------- | --------- | ------ | --------- | --------- | ------- | --------- | -------- | ------- | -------- |
| 72.0444 | 77.4554 | 51.2471 | 64.9538 | 64.0453 | 77.0387 | 85.3310 | 90.83699 | 77.31017 | 78.106886 | 56.103059 | 68.776 | 68.140988 | 72.081567 | 58.6135 | 10.855397 | 67.05879 | 39.2845 | 38.78675 |

**3). DOTA-v2.0**

| Dataset   | **Configs**                                                  | Download                                                     | AP_50 | Backbone | lr schd | bs               |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- | -------- | ------- | ---------------- |
| DOTA-v2.0 | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dotav2.0) | 54.27 | R50      | 12epoch | 2img*2 rtx2080ti |

由于论文长度的限制，论文中没有DOTA-2.0的所有类别的AP，这里列出：

| PL      | BD       | BR         | GTF       | SV        | LV       | SH        | TC      | BC      | ST      | SBF     | RA        | HA      | SP       | HC       | CC     | airport  | helipad | AP50      | AP75       | mAP        |
| ------- | -------- | ---------- | --------- | --------- | -------- | --------- | ------- | ------- | ------- | ------- | --------- | ------- | -------- | -------- | ------ | -------- | ------- | --------- | ---------- | ---------- |
| 76.7619 | 51.55655 | 42.3872759 | 60.464159 | 56.482355 | 55.43076 | 66.681058 | 78.6341 | 60.0626 | 69.6894 | 35.0316 | 56.015956 | 51.9962 | 56.20235 | 54.95597 | 24.335 | 67.31572 | 12.9641 | 54.266644 | 28.8561385 | 30.0281367 |

**4). DIOR-R**

| Dataset | **Configs**                                                  | Download                                                     | AP_50 | Backbone | lr schd | bs               |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- | -------- | ------- | ---------------- |
| DIOR-R  | [orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_1x_dior) | 67.28 | R50      | 12epoch | 2img*2 rtx2080ti |
| DIOR-R  | [orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior) | 68.84 | Swin-T   | 12epoch | 2img*2 rtx2080ti |
| DIOR-R  | [orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_lsk_t_q300_layer2_head64_point32_1x_dior) | 65.07 | LSK-Net  | 12epoch | 2img*2 rtx2080ti |

**5). ICDAR-2015**

| Dataset   | **Configs**                                                  | Download                                                     | P  | R | F-measure  | Backbone | lr schd | bs               |
| --------- | --------------------------------|-----------|----------------- | ------------------------------------------------------------ | ---- | -------- | ------- | ---------------- |
| ICDAR2015 | [orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py](https://github.com/wokaikaixinxin/OrientedFormer/blob/main/projects/OrientedFormer/configs/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015.py) | [Hugging Face](https://huggingface.co/wokaikaixinxin/OrientedFormer/tree/main/orientedformer_le90_r50_q300_layer2_head64_point32_2x_icdar2015) | 85.3 | 74.2  | 79.4 | R50 |24epoch|2img*2 rtx2080ti|



## 引用 OrientedFormer

**“与其引用不开源代码、假开源代码的文章，不如引用真开源代码的文章。”**

```
@ARTICLE{10669376,
  author={Zhao, Jiaqi and Ding, Zeyu and Zhou, Yong and Zhu, Hancheng and Du, Wen-Liang and Yao, Rui and El Saddik, Abdulmotaleb},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={OrientedFormer: An End-to-End Transformer-Based Oriented Object Detector in Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-16},
  keywords={Encoding;Object detection;Proposals;Detectors;Remote sensing;Current transformers;Position measurement;End-to-end detectors;oriented object detection;positional encoding (PE);remote sensing;transformer},
  doi={10.1109/TGRS.2024.3456240}}
```


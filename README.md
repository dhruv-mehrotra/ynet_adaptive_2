# Adaptive Y-Net from a Causal Representation Perspective

**[`Paper`](https://arxiv.org/abs/2111.14820) | [`Video`](https://youtu.be/mqx988tyhfc) | [`Spurious`](https://github.com/vita-epfl/causalmotion/tree/main/spurious) | [`Style`](https://github.com/vita-epfl/causalmotion/tree/main/style)**

This is an addition to the [official implementation](https://github.com/vita-epfl/causalmotion) for the paper

**Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective**
<br>
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022.*
<br>
<a href="https://sites.google.com/view/yuejiangliu">Yuejiang Liu</a>,
<a href="https://www.riccardocadei.com">Riccardo Cadei</a>,
<a href="https://people.epfl.ch/jonas.schweizer/?lang=en">Jonas Schweizer</a>,
<a href="https://www.linkedin.com/in/sherwin-bahmani-a2b5691a9">Sherwin Bahmani</a>,
<a href="https://people.epfl.ch/alexandre.alahi/?lang=en/">Alexandre Alahi</a>
<br>
École Polytechnique Fédérale de Lausanne (EPFL)

TL;DR: incorporate causal *invariance* and *structure* into the design and training of motion forecasting models to improve the *robustness* and *reusability* of the learned representations under common distribution shifts
* causal formalism of motion forecasting with three groups of latent variables
* causal (invariant) representations to suppress spurious features and promote robust generalization
* causal (modular) structure to approximate a sparse causal graph and facilitate efficient adaptation

<p align="left">
  <img src="docs/overview.png" width="800">
</p>

If you find this code useful for your research, please cite our paper:

```bibtex
@InProceedings{Liu2022CausalMotionRepresentations,
    title     = {Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective},
    author    = {Liu, Yuejiang and Cadei, Riccardo and Schweizer, Jonas and Bahmani, Sherwin and Alahi, Alexandre},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17081-17092}
}
```

### Setup

Install PyTorch, for example using pip

```
pip install --upgrade pip
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install dependencies
```
pip install -r requirements.txt
```

Build [ddf dependency](https://github.com/theFoxofSky/ddfnet)
```
cd ddf
python setup.py install
mv build/lib*/* .
```

### Dataset

Get the raw dataset, our filtered custom dataset and segmentation masks for SDD from the original Y-net authors
```
pip install gdown && gdown https://drive.google.com/uc?id=14Jn8HsI-MjNIwepksgW4b5QoRcQe97Lg
unzip sdd_ynet.zip
```

After unzipping the file the directory should have following structure:
```
sdd_ynet
├── dataset_raw
├── dataset_filter
│   ├── dataset_ped
│   ├── dataset_biker
│   │   ├── gap
│   │   └── no_gap
│   └── ...
└── ynet_additional_files
```

In addition to our custom datasets in sdd_ynet/dataset_filter, you can create custom datasets:
```
bash create_custom_dataset.sh
```

### Scripts

1. Train Baseline

```
bash run_train.sh
```

&nbsp;&nbsp;&nbsp;&nbsp;Our pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1HzHP2_Mg2bAlDV3bQERoGQU3PvijKQmU).
```
cd ckpts
gdown https://drive.google.com/uc?id=180sMpRiGhZOyCaGMMakZPXsTS7Affhuf
```

<!-- 2. Zero-shot Evaluation

```
bash run_eval.sh
``` -->

2. Low-shot Adaptation

```
bash run_vanilla.sh
```

```
bash run_encoder.sh
```

```
python utils/visualize.py 
```

### Basic Results

Results of different methods for low-shot transfer across agent types and speed limits.

<img src="docs/result.png" height="180"/>

### Acknowledgement

Out code is developed upon the public code of [Y-net](https://github.com/HarshayuGirase/Human-Path-Prediction/tree/master/ynet) and [Decoupled Dynamic Filter](https://github.com/theFoxofSky/ddfnet).

# queenvis: A Minimal Video Instance Segmentation Framework without Video-based Training

[De-An Huang](https://ai.stanford.edu/~dahuang/), [Zhiding Yu](https://chrisding.github.io/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/)

[[`arXiv`](https://arxiv.org/abs/2208.02245)] [[`Project`]()] [[`BibTeX`](#Citingqueenvis)]

<div align="center">
  <img src="https://ai.stanford.edu/~dahuang/images/queenvis.png" width="100%" height="100%"/>
</div>

### Features
* Video instance segmentation by only training an image instance segmentation model.
* Support major video instance segmentation datasets: YouTubeVIS 2019/2021, Occluded VIS (OVIS).

### Qualitative Results on Occluded VIS
<img src="https://ai.stanford.edu/~dahuang/images/ovis_sheep.gif" height="200"/> <img src="https://ai.stanford.edu/~dahuang/images/ovis_fish.gif" height="200"/>

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for queenvis](datasets/README.md).

See [Getting Started with queenvis](GETTING_STARTED.md).

## Model Zoo

Trained models are available for download in the [queenvis Model Zoo](MODEL_ZOO.md).

## License

The majority of queenvis is made available under the [Nvidia Source Code License-NC](LICENSE). The trained models in the [queenvis Model Zoo](MODEL_ZOO.md) are made available under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Portions of the project are available under separate license terms: Mask2Former is licensed under a [MIT License](https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE). Swin-Transformer-Semantic-Segmentation is licensed under the [MIT License](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="Citingqueenvis"></a>Citing queenvis

```BibTeX
@inproceedings{huang2022queenvis,
  title={queenvis: A Minimal Video Instance Segmentation Framework without Video-based Training},
  author={De-An Huang and Zhiding Yu and Anima Anandkumar},
  journal={NeurIPS},
  year={2022}
}
```

## Acknowledgement

This repo is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).

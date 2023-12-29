# Long-Range-Grouping-Transformer
Official PyTorch implementation of the paper: 

Long-Range Grouping Transformer for Multi-View 3D Reconstruction

**Authors**: Liying Yang, Zhenwei Zhu, Xuxin Lin, Jian Nong, Yanyan Liang.

<img src="./imgs/LGA.gif" width="300"/> <img src="./imgs/SGA.gif" width="300"/>
<img src="./imgs/FRA.gif" width="300"/> <img src="./imgs/TGA.gif" width="300"/>



## Performance

| Methods                                                      | 1 view          | 2 views         | 3 views         | 4 views         | 5 views         | 8 views         | 12 views        | 16 views        | 20 views        |
| ------------------------------------------------------------ | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| [3D-R2N2](https://github.com/chrischoy/3D-R2N2)              | 0.560 / 0.351   | 0.603 / 0.368   | 0.617 / 0.372   | 0.625 / 0.378   | 0.634 / 0.382   | 0.635 / 0.383   | 0.636 / 0.382   | 0.636 / 0.382   | 0.636 / 0.383   |
| [AttSets](https://github.com/Yang7879/AttSets)               | 0.642 / 0.395   | 0.662 / 0.418   | 0.670 / 0.426   | 0.675 / 0.430   | 0.677 / 0.432   | 0.685 / 0.444   | 0.688 / 0.445   | 0.692 / 0.447   | 0.693 / 0.448   |
| [Pix2Vox++](https://github.com/hzxie/Pix2Vox)                | 0.670 / 0.436   | 0.695 / 0.452   | 0.704 / 0.455   | 0.708 / 0.457   | 0.711 / 0.458   | 0.715 / 0.459   | 0.717 / 0.460   | 0.718 / 0.461   | 0.719 / 0.462   |
| [GARNet](https://github.com/GaryZhu1996/GARNet)              | 0.673 / 0.418   | 0.705 / 0.455   | 0.716 / 0.468   | 0.722 / 0.475   | 0.726 / 0.479   | 0.731 / 0.486   | 0.734 / 0.489   | 0.736 / 0.491   | 0.737 / 0.492   |
| [GARNet+](https://github.com/GaryZhu1996/GARNet)             | 0.655 / 0.399   | 0.696 / 0.446   | 0.712 / 0.465   | 0.719 / 0.475   | 0.725 / 0.481   | 0.733 / 0.491   | 0.737 / 0.498   | 0.740 / 0.501   | 0.742 / 0.504   |
| [EVolT](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Multi-View_3D_Reconstruction_With_Transformers_ICCV_2021_paper.pdf) | - / -           | - / -           | - / -           | 0.609 / 0.358   | - / -           | 0.698 / 0.448   | 0.720 / 0.475   | 0.729 / 0.486   | 0.735 / 0.492   |
| [LegoFormer](https://github.com/faridyagubbayli/LegoFormer)  | 0.519 / 0.282   | 0.644 / 0.392   | 0.679 / 0.428   | 0.694 / 0.444   | 0.703 / 0.453   | 0.713 / 0.464   | 0.717 / 0.470   | 0.719 / 0.472   | 0.721 / 0.472   |
| [3D-C2FT](https://github.com/bluestyle97/awesome-3d-reconstruction-papers) | 0.629 / 0.371   | 0.678 / 0.424   | 0.695 / 0.443   | 0.702 / 0.452   | 0.702 / 0.458   | 0.716 / 0.468   | 0.720 / 0.475   | 0.723 / 0.477   | 0.724 / 0.479   |
| [3D-RETR <br> <font size=2>(3 view)</font>](https://github.com/fomalhautb/3D-RETR) | 0.674 / -       | 0.707 / -       | 0.716 / -       | 0.720 / -       | 0.723 / -       | 0.727 / -       | 0.729 / -       | 0.730 / -       | 0.731 / -       |
| [3D-RETR*](https://github.com/fomalhautb/3D-RETR)            | 0.680 / -       | 0.701 / -       | 0.716 / -       | 0.725 / -       | 0.736 / -       | 0.739 / -       | 0.747 / -       | 0.755 / -       | 0.757 / -       |
| [UMIFormer](https://github.com/GaryZhu1996/UMIFormer)        | 0.6802 / 0.4281 | 0.7384 / 0.4919 | 0.7518 / 0.5067 | 0.7573 / 0.5127 | 0.7612 / 0.5168 | 0.7661 / 0.5213 | 0.7682 / 0.5232 | 0.7696 / 0.5245 | 0.7702 / 0.5251 |
| [UMIFormer+](https://github.com/GaryZhu1996/UMIFormer)       | 0.5672 / 0.3177 | 0.7115 / 0.4568 | 0.7447 / 0.4947 | 0.7588 / 0.5104 | 0.7681 / 0.5216 | 0.7790 / 0.5348 | 0.7843 / 0.5415 | 0.7873 / 0.5451 | 0.7886 / 0.5466 |
| LRGT (Ours)                                                  | 0.6962 / 0.4461 | 0.7462 / 0.5005 | 0.7590 / 0.5148 | 0.7653 / 0.5214 | 0.7692 / 0.5257 | 0.7744 / 0.5311 | 0.7766 / 0.5337 | 0.7781 / 0.5347 | 0.7786 / 0.5353 |
| LRGT+ (Ours)                                                 | 0.5847 / 0.3378 | 0.7145 / 0.4618 | 0.7476 / 0.4989 | 0.7625 / 0.5161 | 0.7719 / 0.5271 | 0.7833 / 0.5403 | 0.7888 / 0.5467 | 0.7912 / 0.5497 | 0.7922 / 0.5510 |

###### * The results in this row are derived from models that train individually for the various number of input views.



## TODO

The code and pretrain models are coming soon.
- [ ] Release the pretrain models
- [x] Release the code

## Installation
The environment was tested on Ubuntu 16.04.5 LTS and Ubuntu 20.04.5 LTS. We trained LRGT on 2 Tesla V100s for about 1 day and LRGT+ on 8 Tesla V100s for about 2.5 days.

Clone the code repository
```
git clone https://github.com/LiyingCV/Long-Range-Grouping-Transformer.git
```

Create a new environment from ```environment.yml```
```
conda env create -f environment.yml
conda activate lrgt
```
Or install Python dependencies
```
cd Long-Range-Grouping-Transformer
conda create -n lrgt python=3.6
pip install -r requirements.txt
```

## Demo
<img src="imgs/visualize.gif" width="900"/>

## Datasets

We use the [ShapeNet](https://www.shapenet.org/) and [Pix3D](http://pix3d.csail.mit.edu/) in our experiments, which are available below:

- ShapeNet rendering images: http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
- ShapeNet voxelized models: http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
- Pix3D images & voxelized models: http://pix3d.csail.mit.edu/data/pix3d.zip

## Get start

### Training
We provide the training script, which you can run as following: `sh train.sh`.

We use `torch.distributed` for multiple GPU training; therefore, you can change `CUDA_VISIBLE_DEVICES` and `nproc_per_node` to use more devices or only one device.
### Evaluation
We provide the testing script, which you can run as following: `sh test.sh`
## Citation

If you find our code or paper useful in your research, please consider citing:
```BibTex
@InProceedings{Yang_2023_ICCV,
    author    = {Yang, Liying and Zhu, Zhenwei and Lin, Xuxin and Nong, Jian and Liang, Yanyan},
    title     = {Long-Range Grouping Transformer for Multi-View 3D Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18257-18267}
}
```

## Futher Information

Please check out other works on multi-view reconstruction from our group:
- [GARNet: Global-Aware Multi-View 3D Reconstruction Network and the Cost-Performance Tradeoff (Pattern Recognition 2023)](https://github.com/GaryZhu1996/GARNet)
- [UMIFormer: Mining the Correlations between Similar Tokens for Multi-View 3D Reconstruction (ICCV 2023)](https://github.com/GaryZhu1996/UMIFormer)

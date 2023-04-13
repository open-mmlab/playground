# SATRN

> [On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention](https://arxiv.org/abs/1910.04396)

<!-- [ALGORITHM] -->

## Abstract

Scene text recognition (STR) is the task of recognizing character sequences in natural scenes. While there have been great advances in STR methods, current methods still fail to recognize texts in arbitrary shapes, such as heavily curved or rotated texts, which are abundant in daily life (e.g. restaurant signs, product labels, company logos, etc). This paper introduces a novel architecture to recognizing texts of arbitrary shapes, named Self-Attention Text Recognition Network (SATRN), which is inspired by the Transformer. SATRN utilizes the self-attention mechanism to describe two-dimensional (2D) spatial dependencies of characters in a scene text image. Exploiting the full-graph propagation of self-attention, SATRN can recognize texts with arbitrary arrangements and large inter-character spacing. As a result, SATRN outperforms existing STR models by a large margin of 5.7 pp on average in "irregular text" benchmarks. We provide empirical analyses that illustrate the inner mechanisms and the extent to which the model is applicable (e.g. rotated and multi-line text). We will open-source the code.

<div align=center>
<img src="https://user-images.githubusercontent.com/22607038/142798828-cc4ded5d-3fb8-478c-9f3e-74edbcf41982.png"/>
</div>

## Dataset

### Train Dataset

| trainset  | instance_num | repeat_num | source |
| :-------: | :----------: | :--------: | :----: |
| SynthText |   7266686    |     1      | synth  |
|  Syn90k   |   8919273    |     1      | synth  |

### Test Dataset

| testset | instance_num |   type    |
| :-----: | :----------: | :-------: |
| IIIT5K  |     3000     |  regular  |
|   SVT   |     647      |  regular  |
|  IC13   |     1015     |  regular  |
|  IC15   |     2077     | irregular |
|  SVTP   |     645      | irregular |
|  CT80   |     288      | irregular |

## Results and Models

|                                Methods                                 |        | Regular Text |           |     |           | Irregular Text |        |                                download                                 |
| :--------------------------------------------------------------------: | :----: | :----------: | :-------: | :-: | :-------: | :------------: | :----: | :---------------------------------------------------------------------: |
|                                                                        | IIIT5K |     SVT      | IC13-1015 |     | IC15-2077 |      SVTP      |  CT80  |                                                                         |
|      [Satrn](/configs/textrecog/satrn/satrn_shallow_5e_st_mj.py)       | 0.9600 |    0.9181    |  0.9606   |     |  0.8045   |     0.8837     | 0.8993 | [model](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_shallow_5e_st_mj/satrn_shallow_5e_st_mj_20220915_152443-5fd04a4c.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_shallow_5e_st_mj/20220915_152443.log) |
|    [Satrn-TTA](/configs/textrecog/satrn/satrn_shallow_5e_st_mj.py)     | 0.9530 |    0.9181    |  0.9527   |     |  0.8276   |     0.8884     | 0.9028 |                                                                         |
| [Satrn_small](/configs/textrecog/satrn/satrn_shallow-small_5e_st_mj.py) | 0.9423 |    0.9011    |  0.9567   |     |  0.7886   |     0.8574     | 0.8472 | [model](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_shallow-small_5e_st_mj/satrn_shallow-small_5e_st_mj_20220915_152442-5591bf27.pth) \| [log](https://download.openmmlab.com/mmocr/textrecog/satrn/satrn_shallow-small_5e_st_mj/20220915_152442.log) |
| [Satrn_small-TTA](/configs/textrecog/satrn/satrn_shallow-small_5e_st_mj.py) | 0.9380 |    0.8995    |  0.9488   |     |  0.8122   |     0.8620     | 0.8507 |                                                                         |

## Citation

```bibtex
@article{junyeop2019recognizing,
  title={On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention},
  author={Junyeop Lee, Sungrae Park, Jeonghun Baek, Seong Joon Oh, Seonghyeon Kim, Hwalsuk Lee},
  year={2019}
}
```

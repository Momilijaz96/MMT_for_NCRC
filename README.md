# MultiModal-Transformers-for-Nurse-Activity-Recognition [[arXiv]](https://arxiv.org/pdf/2204.04564.pdf)
This repo is a placeholder for official implementation of the paper "Multimodal transformer for Nurse Activity Recognition", published in CVPM2020, CVPRW.

# Introduction

  This paper proposes a novel transformer based real world action recognition method. The proposed method involves two single modality transformer models, for performing action recogniton on [Nurse-Activity-Recogntion-dataset(2019)](https://ieee-dataport.org/competitions/nurse-care-activity-recognition-challenge). First single moadlity transformer extract sptio-temporal features from skeletal joints of data the subjects and tries to recognize nurse activities from just single modality data. Second single modality transformer performs action recogniton by modeling correlation between acceleration of the performer. Both models are shwon as follows. 

| <img src="https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/single.png " width="900"> | 
|:--:| 
| *Single Modality Transformers (a) Skeletal Joints Model (b) Acceleraion Model* |

We propose a multi-modal transformer by combining both skeletal joints and acceleration data models' final cls tokens and also introuce an additional cross view fusion between both model's layer to develop stronger and better feature vectors for final action recognition. In fusion layer, the spatio-temporal skeletal joints tokens attend to the self-encoded acceleration tokens, which is repeated in all layers. Our result deonstrate the fusing acceleration and skeletal joints gives better action recogniton performance as compare to single modality transformers and simple fusion of both models wiithout cross view fusion. 
| ![alt text](https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/fusion.png) | 
|:--:| 
| *Cross View Fusion Model (a) Cross View Fusion (b) MultiModal Transformer with CrossView Fusion * |




# Results and Models
# Usage
## Citation
If you find this useful in your work, please consider citing,
```
@article{momal2022multimodal_transformer,
  title={Multimodal Transformer for Nurse Activity Recognition},
  author={Momal Ijaz, Renato Diaz ,Chen Chen},
  journal={arXiv preprint arXiv:2204.04564},
  year={2022}}
```

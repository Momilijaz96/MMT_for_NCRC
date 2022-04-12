# MultiModal-Transformers-for-Nurse-Activity-Recognition [[arXiv]](https://arxiv.org/pdf/2204.04564.pdf)
This repo is a placeholder for official implementation of the paper "Multimodal transformer for Nurse Activity Recognition", published in CVPM2020, CVPRW.

# Introduction
This paper proposes a novel transformer based real world action recognition method. The proposed method involves two single modality transformer models, for performing action recogniton on [Nurse-Activity-Recogntion-dataset(2019)](https://ieee-dataport.org/competitions/nurse-care-activity-recognition-challenge). First single moadlity transformer extract sptio-temporal features from skeletal joints of data the subjects and tries to recognize nurse activities from just single modality data. Second single modality transformer performs action recogniton by modeling correlation between acceleration of the performer. Both models are shwon as follows. 
| ![alt text](https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/single.png) | 
|:--:| 
| *Single Modality Transformers (a) Skeletal Joints Model (b) Acceleraion Model* |



This paper proposes a novel transformer based real world action recognition method. The proposed method involves two single modality transformer models, for performing action recogniton on [Nurse-Activity-Recogntion-dataset(2019)](https://ieee-dataport.org/competitions/nurse-care-activity-recognition-challenge). First single moadlity transformer extract sptio-temporal features from skeletal joints of data the subjects and tries to recognize nurse activities from just single modality data. Second single modality transformer performs action recogniton by modeling correlation between acceleration of the performer. Both models are shwon as follows. 

![alt text](https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/fusion.png)


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

# MultiModal-Transformers-for-Nurse-Activity-Recognition [[arXiv]](https://arxiv.org/pdf/2204.04564.pdf)
This repo is for official implementation of the paper "Multimodal transformer for Nurse Activity Recognition", published in CVPM2020, CVPRW.

# Introduction

  This paper proposes a novel transformer based real world action recognition method. The proposed method involves two single modality transformer models, for performing action recogniton on [Nurse-Activity-Recogntion-dataset(2019)](https://ieee-dataport.org/competitions/nurse-care-activity-recognition-challenge). First single moadlity transformer extract sptio-temporal features from skeletal joints of data the subjects and tries to recognize nurse activities from just single modality data. Second single modality transformer performs action recogniton by modeling correlation between acceleration of the performer. Both models are shwon as follows. 

| <img src="https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/single.png "> | 
|:--:| 
| *Single Modality Transformers (a) Skeletal Joints Model (b) Acceleraion Model* |

We propose a multi-modal transformer by combining both skeletal joints and acceleration data models' final cls tokens and also introuce an additional cross view fusion between both model's layer to develop stronger and better feature vectors for final action recognition. In fusion layer, the spatio-temporal skeletal joints tokens attend to the self-encoded acceleration tokens, which is repeated in all layers. Our result deonstrate the fusing acceleration and skeletal joints gives better action recogniton performance as compare to single modality transformers and simple fusion of both models wiithout cross view fusion. 
| ![alt text](https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/fusion.png) | 
|:--:| 
| *Cross View Fusion Model (a) Cross View Fusion (b) MultiModal Transformer with CrossView Fusion * |


# Results and Checkpoints
| Model                     | Accuracy | F1-score  | Precision |  Recall | CheckPoint|
| ------------------------- |:--------:| ---------:| ---------:| -------:| ---------:|
| Skeleton Model            |   76.7   |   67.0    |   69.1    |   70.5  | [SkeletonModel.pth](https://drive.google.com/file/d/1vUMj_7Xjkc5IurVi6FS66IXj5dfJSnAq/view?usp=sharing)
| Acceleration Model        |   45.6   |   10.9    |   9.3     |   14.9  | [AccModel.pth](https://drive.google.com/file/d/16ROhR6_thVaj-1dqSN-hKJSH5fRAYkVL/view?usp=sharing)
| Simple Fusion             |   75.0   |   71.6    |   75.6    |   72.3  | [SimpleFusion.pth](https://drive.google.com/file/d/1HNYp4HAU3mpUzikxkf_uSkcyz7kwLQK4/view?usp=sharing)
| Cross View Fusion Model   |   81.8   |   78.4.   |   79.4    |   78.3  | [CrossViewFusion.pth](https://drive.google.com/file/d/1SWQ3EbLvH_hauJE22eqrYatsqv2e4rAO/view?usp=sharing)

## Comparison with state-of-the-art
| Sensors Used                           |    Method     | Validation Accuracy |
| ---------------------------------------|:-------------:| -------------------:|
| Motion Capture and Location            |      [KNN](https://dl.acm.org/doi/pdf/10.1145/3341162.3344859)      |        80.2         |
| Motion Capture                         |     [ST-GCN](https://dl.acm.org/doi/abs/10.1145/3341162.3345581)    |        64.6         |
| All Modalities                         |      [CNN](https://www.researchgate.net/publication/335765627_Nurse_care_activity_recognition_challenge_summary_and_results)      |        46.5         |
| Acceleration                           | [Random Forest](https://www.researchgate.net/publication/335765627_Nurse_care_activity_recognition_challenge_summary_and_results) |        43.1         |
| Motion Capture and Location            |      [GRU](https://dl.acm.org/doi/abs/10.1145/3341162.3344848)      |        29.3         |
| Acceleration and Motion Capture (Ours) |  [Transformers](https://arxiv.org/pdf/2204.04564.pdf) |        81.8         |

Class Wise F1-score comparison with top two solutions posted for the nurse Activity Recogniton challenge dataset, STGCN and KNN is as follows. we can see for almost all classes our proposed solution out-performs the ST-GCN and hand-crafted feature based KNN method.

<img src="https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/f1.png " width="400"/> 

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

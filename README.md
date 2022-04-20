# MultiModal-Transformers-for-Nurse-Activity-Recognition [[arXiv]](https://arxiv.org/pdf/2204.04564.pdf)
This repo is for official implementation of the paper "Multimodal transformer for Nurse Activity Recognition", published in the Fifth International Workshop on Computer Vision for Physiological Measurement (CVPM), in conjunction with CVPR 2022.

# Introduction

  This paper proposes a novel transformer based real world action recognition method. The proposed method involves two single modality transformer models, for performing action recogniton on [Nurse-Activity-Recogntion-dataset(2019)](https://ieee-dataport.org/competitions/nurse-care-activity-recognition-challenge). First single moadlity transformer extract sptio-temporal features from skeletal joints of data the subjects and tries to recognize nurse activities from just single modality data. Second single modality transformer performs action recogniton by modeling correlation between acceleration of the performer. Both models are shwon as follows. 

| <img src="https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/single.png "> | 
|:--:| 
| __Single Modality Transformers (a) Skeletal Joints Model (b) Acceleraion Model__ |

We propose a multi-modal transformer by combining both skeletal joints and acceleration data models' final cls tokens and also introuce an additional cross view fusion between both model's layer to develop stronger and better feature vectors for final action recognition. In fusion layer, the spatio-temporal skeletal joints tokens attend to the self-encoded acceleration tokens, which is repeated in all layers. Our result deonstrate the fusing acceleration and skeletal joints gives better action recogniton performance as compare to single modality transformers and simple fusion of both models wiithout cross view fusion. 
| ![alt text](https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/fusion.png) | 
|:--:| 
| __Cross View Fusion Model (a) Cross View Fusion (b) MultiModal Transformer with CrossView Fusion__ |


# Results and Checkpoints
| Model                     | Accuracy | F1-score  | Precision |  Recall | CheckPoint|
| ------------------------- |:--------:| ---------:| ---------:| -------:| ---------:|
| Skeleton Model            |   76.7   |   67.0    |   69.1    |   70.5  | [SkeletonModel.pth](https://drive.google.com/file/d/1vUMj_7Xjkc5IurVi6FS66IXj5dfJSnAq/view?usp=sharing)
| Acceleration Model        |   45.6   |   10.9    |   9.3     |   14.9  | [AccModel.pth](https://drive.google.com/file/d/16ROhR6_thVaj-1dqSN-hKJSH5fRAYkVL/view?usp=sharing)
| Simple Fusion             |   75.0   |   71.6    |   75.6    |   72.3  | [SimpleFusion.pth](https://drive.google.com/file/d/1HNYp4HAU3mpUzikxkf_uSkcyz7kwLQK4/view?usp=sharing)
| Cross View Fusion Model   |   81.8   |   78.4.   |   79.4    |   78.3  | [CrossViewFusion.pth](https://drive.google.com/file/d/1SWQ3EbLvH_hauJE22eqrYatsqv2e4rAO/view?usp=sharing)

## Comparison with state-of-the-art
We compare our methods with all other existing solutions reported on the NCRC dataset, including the hand-crafted-feature-based KNN winning entry. NCRC dataset offers three different sensors data during course of performing action, including
* Motion Capture - 29 Skeletal joints data of nurse
* Acceleration - Acceleration of the nurse
* Location - (x,y) location of the nurse 
Table below lists results for different methods utilizing different modalities, whereas our transformer based solution outperforms them all.

| Sensors Used                           |    Method     | Validation Accuracy |
| ---------------------------------------|:-------------:| -------------------:|
| __Acceleration and Motion Capture (Ours)__| __[Transformers](https://arxiv.org/pdf/2204.04564.pdf)__ |      __81.8__         |
| Motion Capture and Location            |      [KNN](https://dl.acm.org/doi/pdf/10.1145/3341162.3344859)      |        80.2         |
| Motion Capture                         |     [ST-GCN](https://dl.acm.org/doi/abs/10.1145/3341162.3345581)    |        64.6         |
| All Modalities                         |      [CNN](https://www.researchgate.net/publication/335765627_Nurse_care_activity_recognition_challenge_summary_and_results)      |        46.5         |
| Acceleration                           | [Random Forest](https://www.researchgate.net/publication/335765627_Nurse_care_activity_recognition_challenge_summary_and_results) |        43.1         |
| Motion Capture and Location            |      [GRU](https://dl.acm.org/doi/abs/10.1145/3341162.3344848)      |        29.3         |

Graphs shown below reflect the effectivness of proposed solution. Pn right, he bar graph shows class Wise F1-score comparison with top two solutions posted for the nurse Activity Recogniton challenge dataset, STGCN and KNN. We can see for almost all classes our proposed solution out-performs the ST-GCN and hand-crafted feature based KNN method. On right, we have validation accuracy for all existing solutions as mentioned in table above. 

<img src="https://github.com/Momilijaz96/MMT_for_NCRC/blob/main/images/results.png" width="700"/> 


# Usage
## Requirements
Create a conda environment and install dependencies from given requirements.txt.
```
conda create --name myenv python=3.6
conda env create -f Tools/mmt_env.yml
```
## Training 
Download the data and put the path of acceleration and skeletal joints data and labels in the config file. Simply run the following command to train the crossview fusion model on the NurseCareActivityRecognition dataset. 
```
python3 train_ncrc.py 
```
Note: For training another model, you can simply import relevant model in train_ncrc script.
## Inference
For inference load desired chcekpoint and select a model name. For example for validation on NCRC data using CrossView fusion model, run.
Where CKTP_PATH is the path to correspoding downloaded checkpoint model, and a valid model name can be 
* crossview_fusion_model
* model_acc_only
* model_skeleton_only
* model_simple_fusion


```
python3 validation_ncrc.py --ckpt_path [CKPT PATH] --model 'crossview_fusion_model'
```
## Citation
If you find this useful in your work, please give a ⭐ and consider citing:
```
@article{momal2022multimodal_transformer,
  title={Multimodal Transformer for Nurse Activity Recognition},
  author={Momal Ijaz, Renato Diaz ,Chen Chen},
  journal={arXiv preprint arXiv:2204.04564},
  year={2022}}
```

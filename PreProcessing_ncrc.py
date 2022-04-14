import os
import pandas as pd
import numpy as np
import re

MOCAP_FREQ = 100
MEDITAG_FREQ = 20
ACC_FREQ = 4

MOCAP_WINDOW= 200 #SAMPLE IN A 2 SEC WINDOW OF MOCAP SENSOR
MEDITAG_WINDOW= 20 #SAMPLES IN A 1 SEC WINDOW OF A MOCAP SENSOR
MOCAP_WINDOWS = 6000//MOCAP_WINDOW #MOCAP sensor can have 30 windows per action of 2 sec
MEDIAG_WINDOWS = 600//MEDITAG_WINDOW #MEDITAG can have 30 windows of size 30 sec per action


#################### UTILITY FUNCTIONS #######################
'''
FUNCTION: to map data for one act to an id and label

Argumetnts::
partition = test/ train
data_path (list): list of path to find specified data modalities
labels_path (string): path to the labels file
aug_path (string): path to the augmented data
start_i (int): int  value to start id of sampels at

Output::
TEST partition
pose2id = {id-10:{mocap: [path,[w1,w2,w3 ... w30]. meditag: [path,[w1,w2,w3 ... w30]},...}
labels = {id=10: 0, id-11: 2, ...}

TRAIN partition:
pose2id = {id-10:{mocap: [path,w1], meditag: [path,w1] },...}
labels2id = {id-0: 0, id-1: 2, ...}
'''

def pose2idlabel(data_path,labels_path,aug_path=None,start_i=0):
    pose2id=dict() #path to sample id
    id2label=dict() #sample id to label
    i=start_i #Sample ids start from 0 or given id
    mocap_path = data_path[0]
    acc_path = data_path[1]
    if mocap_path and acc_path:
        
        samples=os.listdir(mocap_path)
        
        for smpl in samples:
            
            mocap_smpl_path=mocap_path+smpl
            
            if mocap_smpl_path.endswith('.csv'):
                
                segment_id = int(re.findall(r'\d+', smpl)[0])
                pose2id['id-'+str(i)] = {} #pose2id['id-0']={'mocap':['segment0.csv',seg_id],'acc':['meditag_train.csv',seg_id]}
                pose2id['id-'+str(i)]['mocap'] = [mocap_smpl_path,segment_id]
                pose2id['id-'+str(i)]['acc'] = [acc_path,segment_id]
                id2label['id-'+str(i)] = get_ncrc_labels(segment_id,labels_path)
                i+=1

    return pose2id,id2label,i


#Function: Label generator for activity recognition for NCRC dataset
#input : segment id
def get_ncrc_labels(segment_id,labels_path):
    lbl_map={2:0, #vital signs measurement
            3:1, #blood collection 
            4:2, #blood glucose measurement
            6:3, #indwelling drop retention and connection
            9:4, #oral care
            12:5} #diaper exchange and cleaning of area
    df=pd.read_csv(labels_path)
    df.drop('subject',axis=1,inplace=True)
    labels_map={}
    for _,row in df.iterrows():
        labels_map[row['segment_id']]=lbl_map[row['activity_id']]
    return labels_map[segment_id]


#################### MAIN ####################
def preprocess():
    #Get pose dir to id dict, and id to label dict

    #NCRC path
    mocap_train_path='/home/mo926312/Documents/falldet/ncrc/train/mocap/'
    mocap_test_path='/home/mo926312/Documents/falldet/ncrc/test/mocap/'

    meditag_train_path='/home/mo926312/Documents/falldet/ncrc/train/meditag_train.csv'
    meditag_test_path='/home/mo926312/Documents/falldet/ncrc/test/meditag_test.csv'

    acc_train_path='/home/mo926312/Documents/falldet/ncrc/train/accelerometer_train.csv'
    acc_test_path='/home/mo926312/Documents/falldet/ncrc/test/accelerometer_test.csv'

    tr_labels_path='/home/mo926312/Documents/falldet/ncrc/train/activities_train.csv'
    tst_labels_path='/home/mo926312/Documents/falldet/ncrc/test/activities_test.csv'


    aug_path=None

    tr_pose2id,tr_labels,start_i = pose2idlabel([mocap_train_path,acc_train_path],tr_labels_path,aug_path)
    pose2id,labels,_ = pose2idlabel([mocap_test_path,acc_test_path],tst_labels_path,aug_path,start_i)

    # For Training #CREATE TEST AND TRAIN IDS
    partition=dict()
    partition['train']=list(tr_pose2id.keys())
    partition['test']=list(pose2id.keys())
    print('--------------DATA SPLIT----------')
    print("Train Sample: ",len(tr_pose2id))
    print("Test Samples: ",len(pose2id))

    #Merge the labels and pose dictionary
    pose2id.update(tr_pose2id)
    labels.update(tr_labels)

    #print("Partition: ",labels)
    print("Partitions are  Made!" )
    return pose2id,labels,partition



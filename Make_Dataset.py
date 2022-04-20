import pandas as pd
import numpy as np
import math
import torch
import random
import torch.nn.functional as F
pd.options.mode.chained_assignment = None  # default='warn'
import scipy.stats as s
import config as cfg

MOCAP_SEGMENT = cfg.data_params['MOCAP_SEGMENT']
ACC_SEGMENT = cfg.data_params['ACC_SEGMENT']


#################### MAIN #####################

#CREATE PYTORCH DATASET
'''
Input Args:
data = ncrc or ntu
num_frames = mocap and nturgb+d frame count!
acc_frames = frames from acc sensor per action
'''

class Poses3d_Dataset(torch.utils.data.Dataset):
    def __init__(self, data, list_IDs, labels, pose2id,  **kwargs): 
        
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.pose2id = pose2id
        self.data=data
        
        if self.data=='ncrc':
            self.partition = kwargs.get('partition', None)
            self.normalize = kwargs.get('normalize', None)
            self.meditag = np.array([])
            self.meditag_segment_id = None
            self.mocap = np.array([])
            self.mocap_segment_id = None
            self.mocap_frames = kwargs.get('mocap_frames', None); self.acc_frames = kwargs.get('acc_frames',None);

        
    #Function to compute magnitude of a signal
    def magnitude(self,data):
        data = data.numpy()
        data[:,0] = data[:,0]**2
        data[:,1] = data[:,1]**2
        data[:,2] = data[:,2]**2
        return np.sqrt(data[:,0]+data[:,1]+data[:,2]).reshape(data.shape[0],1)

    #Function to select frames
    def frame_selection(self,data_sample,num_frames,skip=2):
        
        if (data_sample.shape[0])>num_frames: 
            data_sample=np.array(data_sample[::skip,:,:])#select every second frame
            if data_sample.shape[0]>num_frames: #If after every second frame, the count is higher
                data_sample=data_sample[:num_frames,:,:]
        
        if data_sample.shape[0]<num_frames:
            diff=num_frames-data_sample.shape[0]
	
            #Select diff random frames
            if diff<=data_sample.shape[0]: #If diff<data_sample, do this!
                if diff<data_sample.shape[0]:
                    sampled_frames=random.sample(range(0,data_sample.shape[0]-1), diff)
                elif diff==data_sample.shape[0]:
                    sampled_frames=np.arange(data_sample.shape[0]) #If data_sample len is half of self.frame_num, just copy every frame
                #Copy sampled frames after the original frame
                
                for f in sampled_frames:
                    data_sample=np.insert(data_sample,f,data_sample[f,:,:],axis=0)
            
            elif diff>data_sample.shape[0]: #else if diff>data_sample len
               #Copy every frame twice and then repeat last frame to complete desired len
                sampled_frames=np.arange(data_sample.shape[0])
            
                #Copy sampled frames after the original frame
                for f in sampled_frames:
                    data_sample=np.insert(data_sample,f,data_sample[f,:,:],axis=0)
            
                #If still less then desired num frames, repeat last frame
                rem_diff=num_frames-data_sample.shape[0]
                last_frame=data_sample[-1,:,:]
                tiled=np.tile(last_frame,(rem_diff,1,1))
                data_sample=np.append(data_sample,tiled,axis=0) #repeat last frame pose for diff times

        return data_sample


    #Read segment - Function to read actiona nd convert it to fx29x3
    #Input: segment csv file path, window id
    #Output: Array: fx29x3
    def read_mocap_segment(self,path):
        df=pd.read_csv(path)
        
        df.drop('time_elapsed',axis=1,inplace=True)
        df.drop('segment_id',axis=1,inplace=True)
        
        df.interpolate(method='linear',axis=0,inplace=True)
        df.fillna(method='bfill',axis=0,inplace=True)
        df.fillna(method='ffill',axis=0,inplace=True)
        df.fillna(value=0,axis=0,inplace=True)
        #print("Nans : ",df.isna().sum().sum())
        #assert df.isna().sum().sum() == 0
        
        data=df.to_numpy()
        frames=data.shape[0]
        data=np.reshape(data,(frames,29,3))

        '''
        #Fix the whole segment count - Repeat last pose!
        if (data.shape[0]<MOCAP_SEGMENT):
            last_pose = data[-1,:,:]
            diff = MOCAP_SEGMENT - data.shape[0]
            tiled = np.tile(last_pose,(diff,1,1))
            data = np.append(data,tiled,axis=0)
        elif data.shape[0]>MOCAP_SEGMENT:
            data=data[:MOCAP_SEGMENT]
        '''
        data = self.frame_selection(data,self.mocap_frames,skip=10)

        if self.normalize:
            data = self.normalize_data(data,np.mean(data),np.std(data))

        return data #600 x 29 x 3


    #Read segment - Function to read acceleration data and convert it to 120 x 3 [x,y,z]
    #Input: segment csv file path, window id
    #Output: Array: 20x3
    def read_acc_segment(self,path,segment_id):
        df = pd.read_csv(path)

        df.drop('time_elapsed',axis=1,inplace=True)
        
        df.interpolate(method='linear',axis=0,inplace=True)
        df.fillna(method='bfill',axis=0,inplace=True); win = 40
        #df['x'] = df['x'].ewm(span=win).mean(); df['y'] = df['y'].ewm(span=win).mean()
        #df['z'] = df['z'].ewm(span=win).mean()
        
        df_segment = df.loc[df['segment_id']==segment_id,:].copy() #Extract meditag data of respective sensor, 600 x 4
	
        df_segment.drop('segment_id',axis=1,inplace=True) #120 x 3
        data=df_segment.to_numpy()
        
        #Adjust if last window has less samples
        if data.shape[0]==0:
            #print("INVALID SEGMENT: ",segment_id,'---',path)
            segment219 =  df.loc[df['segment_id'] == 219,:] #Extract segment 219, which is same act performed by same subject as missing one!
            segment219.drop('segment_id',axis=1,inplace=True)
            data219 = segment219.to_numpy()

            segment685 =  df.loc[df['segment_id'] == 685,:] #Another similar segment
            segment685.drop('segment_id',axis=1,inplace=True)
            data685 = segment685.to_numpy()
            
            min_samples = data685.shape[0] if data685.shape[0]<data219.shape[0] else data219.shape[0]
            data219 = data219[:min_samples]
            data685 = data685[:min_samples]

            data = np.mean([data219,data685], axis=0)

        if (data.shape[0]<self.acc_frames):
            last_loc = data[-1,:]
            diff = self.acc_frames - data.shape[0]
            tiled = np.tile(last_loc,(diff,1))
            data = np.append(data,tiled,axis=0)
        elif data.shape[0]>self.acc_frames:
            data=data[:self.acc_frames,:]
        
        if self.normalize:
            data = self.normalize_data(data,np.mean(data),np.std(data))
        
        return data #120 x 3

    def extract_acc_features(self,data):
        #[mean(xyz), std(xyz), Max(xyz), Min(xyz), Kurtosis(xyz), Skewness(xyz)]
        data = data.numpy()
        acc_features=torch.zeros((18,4))
        #Mean
        acc_features[0,0] = np.mean(data[:,0])
        acc_features[1,0] = np.mean(data[:,1])
        acc_features[2,0] = np.mean(data[:,2])

        #Std
        acc_features[3,0] = np.std(data[:,0])
        acc_features[4,0] = np.std(data[:,1])
        acc_features[5,0] = np.std(data[:,2])

        #Max
        acc_features[6,0] = np.max(data[:,0])
        acc_features[7,0] = np.max(data[:,1])
        acc_features[8,0] = np.max(data[:,2])

        #Min
        acc_features[9,0] = np.max(data[:,0])
        acc_features[10,0] = np.max(data[:,1])
        acc_features[11,0] = np.max(data[:,2])

        #Kurtosis
        acc_features[12,0] = s.kurtosis(data[:,0], fisher=False)
        acc_features[13,0] = s.kurtosis(data[:,1], fisher=False)
        acc_features[14,0] = s.kurtosis(data[:,2], fisher=False)

        #Skewness
        acc_features[15,0] = s.skew(data[:,0])
        acc_features[16,0] = s.skew(data[:,1])
        acc_features[17,0] = s.skew(data[:,2])

        return acc_features

    def normalize_data(self,data,mean,std):
        return (data-mean) / std

  
    #Function to get poses for F frames/ one sample, given sample id 
    def get_pose_data(self,id):
        
        if self.data =='ncrc':
         
            segment_info = self.pose2id[id] #get info about paths to action sample
            
            mocap_info = segment_info['mocap'] #get path to one windowed sample mocap - [path,int id]
            acc_info = segment_info['acc'] #get path to one windowed sample meditag - [path, int id]
            
            
            #Extrat segment ids
            segment_id = mocap_info.pop()
            segment_id = acc_info.pop()
            
            mocap_sig = torch.tensor( self.read_mocap_segment( mocap_info[0] ) ) #600 x 29 x 3
            acc_sig = torch.tensor( self.read_acc_segment( acc_info[0] , segment_id ) ) #acc_frames x 3

            #Extract acceleration features
            acc_features = self.extract_acc_features(acc_sig) #ACC_FEATURES x 4
    
            #Add magnitude signal to acceleration
            acc_mag = torch.from_numpy(self.magnitude(acc_sig))
            acc_sig = torch.cat((acc_sig,acc_mag), dim=1) #acc_frames x 4

            ######## Combine the windows of both signals #######

            #concate features and acc data
            acc_data = torch.cat((acc_features,acc_sig),dim=0) #acc_framesx4, ACC_FEATURESx4 -> acc_frames+ACC_FEATURES x 4
            acc_ext = torch.zeros((self.mocap_frames, self.acc_frames + ACC_FEATURES, 4))
            acc_ext[0,:,:] = acc_data #600 x 150+18 x 4

            #add additional dim to mocap_data
            mocap_ext = torch.zeros((self.mocap_frames,29,4))
            mocap_ext[:,:,:3] = mocap_sig

            data_sample = torch.cat((mocap_ext,acc_ext),dim=1) #600x29x3 + 600x162x3 = 200 x 191 x 3
            
            return data_sample #mocap_frames x ACC_FEATUTES+acc_frames+num_joints x 3 = 191


    def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]
            # Load data and get label
            data=self.get_pose_data(ID)
            if isinstance(data,np.ndarray):
                X = torch.from_numpy(data)
            else:
                X = data
            y = self.labels[ID]
            return X, y


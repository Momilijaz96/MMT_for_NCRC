'''
This is simple fusion model, that uses both acceleration and skeletal joints data,
computes their features using single modality models 
and simply concats the final cls tokens of each model before performing final action recogniton.
'''

import math
import logging
from functools import partial
from collections import OrderedDict
from this import s
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import Block


class ActRecogTransformer(nn.Module):
    def __init__(self, device='cpu',  mocap_frames=600, acc_frames=150, num_joints=29, in_chans=3, acc_coords=3, acc_features=18, spatial_embed=32, sdepth=4,adepth=4,tdepth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, op_type='cls', embed_type='lin',fuse_acc_features=False,
                 drop_rate=0.05, attn_drop_rate=0.05, drop_path_rate=0.2,  norm_layer=None, num_classes=6):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            mocap_frames (int): input frame number for skeletal joints
            acc_frames (int): input num frames for acc sensor
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            acc_coords(int): number of coords in one acc reading from meditag sensor: (x,y,z)=3
            spatial_embed (int): spatial patch embedding dimension 
            sdepth (int): depth of spatial  transformer
            tdepth (int): depth of temporal transformer
            adepth (int): depth of acc transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            op_type(string): 'cls' or 'gap', output of temporal and acc encoder is cls token or global avg pool of encoded features.
            embed_type(string): convolutional 'conv' or linear 'lin'
            fuse_acc_features(bool): Wether to fuse acceleration feature into the acc feature or not!

        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) #Momil:embed_dim_ratio is spatial transformer's patch embed size
        temp_embed = spatial_embed*(num_joints)   #### temporal embed_dim is spatial embed*(num_jonits) - as one frame is of this dim! and v have acc_frames + sp_frames
        temp_frames = mocap_frames  #Input frames to the temporal transformer are frames from mocap sensor!
        acc_embed = temp_embed #Since both signals needs to be concatenated, their dm is similar
        self.op_type = op_type
        self.embed_type = embed_type
        
        print("-------------ACCELERATION-------------")
        print("Acc Frames: ",acc_frames)
        print("Acc embed dim: ",acc_embed)
        print("Acc depth: ",adepth)
        
        print('-----------SKELETON---------------')
        print("Temporal input tokens (Frames): ",mocap_frames)
        print("Spatial input tokens (Joints): ",num_joints)
        print("Spatial embed dim: ",spatial_embed)
        print("Temporal embed dim: ",temp_embed)
        print("Spatial depth: ",sdepth)
        print("Temporal depth: ",tdepth)

        print('-------------Regularization-----------')
        print("Drop Rate: ",drop_rate)
        print("Attn drop rate: ",attn_drop_rate) 
        print("Drop path rate: ",drop_path_rate)
        
        #Spatial patch and pos embeddings
        if embed_type=='lin':
            self.Spatial_patch_to_embedding = nn.Linear(in_chans, spatial_embed)#Linear patch embedding
        else:
            self.Spatial_patch_to_embedding = nn.Conv1d(in_chans, spatial_embed, 1, 1)#Conv patch embedding
        
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints+1, spatial_embed))
        self.spat_token = nn.Parameter(torch.zeros(1,1,spatial_embed))
        self.proj_up_clstoken = nn.Linear(mocap_frames*spatial_embed, num_joints*spatial_embed)
        self.sdepth = sdepth
        self.num_joints = num_joints
        self.joint_coords = in_chans

        #Temporal embedding
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, temp_frames+1, temp_embed)) #additional pos embedding zero for class token
        self.temp_frames = mocap_frames

        #Acceleration patch and pos embeddings
        if embed_type=='lin':
            self.Acc_coords_to_embedding = nn.Linear(acc_coords, acc_embed) #Linear patch embedding
        else:
            self.Acc_coords_to_embedding = nn.Conv1d(acc_coords, acc_embed, 1, 1) #Conv patch embedding
        
        self.Acc_pos_embed = nn.Parameter(torch.zeros(1, acc_frames+1, acc_embed)) #1 location per frame - embed: 1xloc_embed from 1xloc_cords
        self.acc_token = nn.Parameter(torch.zeros(1,1,acc_embed))
        self.acc_frames = acc_frames
        self.acc_coords= acc_coords
        self.fuse_acc_features = fuse_acc_features
        self.acc_features = acc_features

        
        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, sdepth)]  #Stochastic depth decay rule
        adpr = [x.item() for x in torch.linspace(0, drop_path_rate, adepth)]  #Stochastic depth decay rule
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]  #Stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(sdepth)])

        self.Acceletaion_blocks = nn.ModuleList([
            Block(
                dim=acc_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=adpr[i], norm_layer=norm_layer)
            for i in range(adepth)])

        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=temp_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])

        self.Spatial_norm = norm_layer(spatial_embed)
        self.Acc_norm = norm_layer(acc_embed)
        self.Temporal_norm = norm_layer(temp_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        #Linear layer to extract features from the acc features signal
        self.acc_features_embed = nn.Linear(acc_features,acc_embed)

        #Classification head
        self.class_head = nn.Sequential(
            nn.LayerNorm(acc_embed+temp_embed),
            nn.Linear(acc_embed+temp_embed, num_classes)
        )

    def Acc_forward_features(self,x):
        b, f, p, c = x.shape  # b is batch size, f is number of frames, c is values per rading 3, p is readig per frames 1, B x Fa X 1 x 3
        
        x = rearrange(x, 'b f p c  -> b f (p c)', ) # b x Fa x 3
        
        if self.embed_type == 'conv':
            x = rearrange(x, '(b f) p c  -> (b f) c p',b=b ) # b x 3 x Fa  - Conv k liye channels first
            x = self.Acc_coords_to_embedding(x) # B x c x p ->  B x Sa x p
            x = rearrange(x, '(b f) Sa p  -> (b f) p Sa', b=b)
        else: 
            x = self.Acc_coords_to_embedding(x) #all acceleration data points for the action = Fa | op: b x Fa x Sa

        
        
        class_token=torch.tile(self.acc_token,(b,1,1)) #(B,1,1) - 1 cls token for all frames

        x = torch.cat((x,class_token),dim=1) 
        _,_,Sa = x.shape
    
        x += self.Acc_pos_embed
        x = self.pos_drop(x)

        for blk in self.Acceletaion_blocks:
            x =blk(x)
            
        x = self.Acc_norm(x)
        
        #Extract cls token
        cls_token = x[:,-1,:]
        if self.op_type=='cls':
            return cls_token
        else:
            x = x[:,:f,:]
            x = rearrange(x, 'b f Sa -> b Sa f')
            x = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x Sa x 1
            x = torch.reshape(x, (b,Sa))
            return x #b x Sa

    def Spatial_forward_features(self, x):
        b, f, p, c = x.shape  # b is batch size, f is number of frames, p is number of joints, c is in_chan 3 
        x = rearrange(x, 'b f p c  -> (b f) p c', ) 

        if self.embed_type == 'conv':
            x = rearrange(x, '(b f) p c  -> (b f) c p',b=b ) # b x 3 x Fa  - Conv k liye channels first
            x = self.Spatial_patch_to_embedding(x) # B x c x p ->  B x Se x p
            x = rearrange(x, '(b f) Se p  -> (b f) p Se', b=b)
        else: 
            x = self.Spatial_patch_to_embedding(x) # B x p x c ->  B x p x Se
        
        class_token=torch.tile(self.spat_token,(b*f,1,1)) #(B,1,1)
        x = torch.cat((x,class_token),dim=1) # b x (p+1) x Se 

        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        
        #Extract cls token
        Se = x.shape[-1]
        cls_token = x[:,-1,:]
        cls_token = torch.reshape(cls_token, (b,f*Se))
        
        #Reshape input
        x = x[:,:p,:]
        x = rearrange(x, '(b f) p Se-> b f (p Se)', f=f)
    
        return x, cls_token #cls token and encoded features returned

    def Temp_forward_features(self, x, cls_token):
        b,f,St = x.shape
        x = torch.cat((x,cls_token), dim=1) #B x mocap_frames +1 x temp_embed | temp_embed = num_joints*Se
        
        b  = x.shape[0]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        
        for blk in self.Temporal_blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        
        ###Extract Class token head from the output
        if self.op_type=='cls':
            cls_token = x[:,-1,:]
            cls_token = cls_token.view(b, -1) # (Batch_size, temp_embed)
            return cls_token

        else:
            x = x[:,:f,:]
            x = rearrange(x, 'b f St -> b St f')
            x = F.avg_pool1d(x,x.shape[-1],stride=x.shape[-1]) #b x St x 1
            x = torch.reshape(x, (b,St))
            return x #b x St


    def forward(self, inputs):
        #Input: B x MOCAP_FRAMES X  119 x 3
        b,_,_,c = inputs.shape

        #Extract skeletal signal from input
        x = inputs[:,:, :self.num_joints, :self.joint_coords] #B x Fs x num_joints x 3
        
        #Extract acc signal from input
        sxf = inputs[:, 0, self.num_joints:self.num_joints+self.acc_features, 0 ] #B x 1 x acc_features x 1
        sx = inputs[:, 0 , self.num_joints+self.acc_features:, :self.acc_coords] #B x 1 x Fa x 3
        sx = torch.reshape(sx, (b,-1,1,self.acc_coords) ) #B x Fa x 1 x 3
        
        #Get skeletal features 
        x,cls_token = self.Spatial_forward_features(x) #in: B x Fs x num_joints x 3 , op: B x Fs x (num_joints*Se)

        #Pass cls token to temporal transformer
        temp_cls_token = self.proj_up_clstoken(cls_token) #in: B x mocap_frames * Se -> op: B x num_joints*Se
        temp_cls_token = torch.unsqueeze(temp_cls_token,dim=1) #op: B x 1 x num_joints*Se
        
        x = self.Temp_forward_features(x,temp_cls_token) #in: B x Fs x (num_joints*Se) , op: B x St

        #Get acceleration features 
        sx = self.Acc_forward_features(sx) #in: F x Fa x 3 x 1,  op: B x St
        sxf = self.acc_features_embed(sxf)
        if self.fuse_acc_features:
            sx+= sxf #Add the features signal to acceleration signal

        #Concat features along frame dimension
        x = torch.cat((x,sx),dim=1) #x += sx 
        x = self.class_head(x)

        return F.log_softmax(x,dim=1)



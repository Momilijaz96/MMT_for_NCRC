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
    def __init__(self, device='cpu', mocap_frames=600, num_joints=29, in_chans=3,  spatial_embed=32, sdepth=4, tdepth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, op_type='cls', embed_type='lin',
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None, num_classes=6):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            mocap_frames (int): input frame number for skeletal joints
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            spatial_embed (int): spatial patch embedding dimension 
            sdepth (int): depth of spatial  transformer
            tdepth (int): depth of temporal transformer
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
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) #Momil:embed_dim_ratio is spatial transformer's patch embed size
        temp_embed = spatial_embed*(num_joints)   #### temporal embed_dim is spatial embed*(num_jonits) - as one frame is of this dim! and v have acc_frames + sp_frames
        temp_frames = mocap_frames  #Input frames to the temporal transformer are frames from mocap sensor!
        acc_embed = temp_embed #Since both signals needs to be concatenated, their dm is similar
        self.op_type = op_type
        self.embed_type = embed_type
        
      
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

        #Temporal embedding
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, temp_frames+1, temp_embed)) #additional pos embedding zero for class token
        self.temp_frames = mocap_frames
        self.joint_coords = in_chans

        
        sdpr = [x.item() for x in torch.linspace(0, drop_path_rate, sdepth)]  #Stochastic depth decay rule
        tdpr = [x.item() for x in torch.linspace(0, drop_path_rate, tdepth)]  #Stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=sdpr[i], norm_layer=norm_layer)
            for i in range(sdepth)])

        self.Temporal_blocks = nn.ModuleList([
            Block(
                dim=temp_embed, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=tdpr[i], norm_layer=norm_layer)
            for i in range(tdepth)])

        self.Spatial_norm = norm_layer(spatial_embed)
        self.Temporal_norm = norm_layer(temp_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)


        #Classification head
        self.class_head = nn.Sequential(
            nn.LayerNorm(temp_embed),
            nn.Linear(temp_embed, num_classes)
        )


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
        #Input: B x MOCAP_FRAMES X  149 x 3
        b,_,_,c = inputs.shape

        #Extract skeletal signal from input
        x = inputs[:,:, :self.num_joints, :self.joint_coords] #B x Fs x num_joints x 3
        
        
        #Get skeletal features 
        x,cls_token = self.Spatial_forward_features(x) #in: B x Fs x num_joints x 3 , op: B x Fs x (num_joints*Se)

        #Pass cls token to temporal transformer
        temp_cls_token = self.proj_up_clstoken(cls_token) #in: B x mocap_frames * Se -> op: B x num_joints*Se
        temp_cls_token = torch.unsqueeze(temp_cls_token,dim=1) #op: B x 1 x num_joints*Se
        
        x = self.Temp_forward_features(x,temp_cls_token) #in: B x Fs x (num_joints*Se) , op: B x St

        #Concat features along frame dimension
        x = self.class_head(x)

        return F.log_softmax(x,dim=1)


'''
model=ActRecogTransformer()
x=torch.randn((14,600,149,3))
op=model(x)
print("Op shape: ",op.shape)
'''

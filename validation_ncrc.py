import numpy as np
from Make_Dataset import Poses3d_Dataset
import PreProcessing_ncrc_losocv
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
#from Model.model_acc_early_fusion import ActRecogTransformer
#from Model.model_acc_only import ActRecogTransformer
#from Model.model_skeleton_only import ActRecogTransformer
from Model.model_acc_qkvfusion import ActRecogTransformer

#CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
print("Creating params....")
params = {'batch_size':8,
          'shuffle': True,
          'num_workers': 3}

# Datasets
pose2id,labels,partition=PreProcessing_ncrc_losocv.preprocess_losocv(5)

print("Creating Data Generators...")
mocap_frames = 600
acc_frames = 150


validation_set = Poses3d_Dataset(data='ncrc',list_IDs=partition['test'], labels=labels, pose2id=pose2id, mocap_frames=mocap_frames,acc_frames=acc_frames, normalize=False)
validation_generator = torch.utils.data.DataLoader(validation_set, **params) #Each produced sample is 6000 x 229 x 3


#Load pretrained model and criterion
model_path = '/home/mo926312/Documents/falldet/Fall-Detection/modelZoo/model_losocv5.pt'
model = ActRecogTransformer(device)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

#Loss
criterion=torch.nn.CrossEntropyLoss()

#Loop over validation split
model.eval()

cnt=0
accuracy=0
all_targets=[]
all_predictions=[]
for batch_idx,sample in enumerate(validation_generator):
    #Transfer to GPU
    inputs, targets = sample
    inputs, targets = inputs.to(device), targets.to(device)

    #Predict fall/no fall activity
    predict_probs=model(inputs.float())
    predict_labels=torch.argmax(predict_probs, 1)

    #Convert to numpy array
    predict_labels = predict_labels.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    
    #Compute number of correctly predicted - Overall
    #prec,rec,f1,_=precision_recall_fscore_support(targets,predict_labels,average='macro')

    #Acompute accuracy
    cnt+=len(targets)
    accuracy += (predict_labels == targets).sum().item()

    all_targets+=list(targets)
    all_predictions+=list(predict_labels)

    #Overall
    #precision.append(prec)
    #recall.append(rec)
    #f1_scores.append(f1)

print("Targets type: ",(all_targets))
print("Prediction type: ",(all_predictions))

prec,rec,f1,_=precision_recall_fscore_support(all_targets,all_predictions,average=None)

print("Precision: ",prec)
print("Recall: ",rec)
print("F1-score: ",f1)
print("Correct out of:",str(accuracy)+'/'+str(cnt))
accuracy *= 100. / cnt
print("Overall Accuracy: ",accuracy)

#Class wise accuracy
matrix = confusion_matrix(all_targets, all_predictions)
print("Class wise Acc: ",matrix.diagonal()/matrix.sum(axis=1))
print(matrix)



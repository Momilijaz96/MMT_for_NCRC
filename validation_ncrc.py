import numpy as np
from Make_Dataset import Poses3d_Dataset
import PreProcessing_ncrc_losocv
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import argparse
import Tools/config as cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train a recognizer')
    parser.add_argument('--ckpt_path', help='Path to ckpt', default = None)
    parser.add_argument('--model', help='Model FileName', default ='model_crossview_fusion')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')



def main():

          args = parse_args()
          
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
          valid_models= ['model_acc_only','model_crossview_fusion','model_skeleton_only','model_simple_fusion']
          model = args.model
          assert model in valid_model, "Please give a valid model name from this list: "+ str(valid_models)
          model_pah = args.ckpt_path
          _temp = __import__('Model.'+model, globals(), locals(), ['ActRecogTransformer'], -1)
          ActRecogTransformer = _temp.ActRecogTransformer

          if model_path is None:
                    print("Please provide model checkpoint as cmd ine arg for --ckpt_path")
                    sys.exis(-1)
          
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
              #prec,rec,f1,_ = precision_recall_fscore_support(targets,predict_labels,average='macro')

              #Acompute accuracy
              cnt+=len(targets)
              accuracy += (predict_labels == targets).sum().item()

              all_targets+=list(targets)
              all_predictions+=list(predict_labels)


          prec,rec,f1,_=precision_recall_fscore_support(all_targets,all_predictions,average=None)
          print("------- CLASS WISE METRICS -----")
          print("Class wise Precision: ",prec)
          print("Class wise Recall: ",rec)
          print("Class wise F1-score: ",f1)
          matrix = confusion_matrix(all_targets, all_predictions)
          print("Class wise Acc: ",matrix.diagonal()/matrix.sum(axis=1))
          print(matrix)

          print("----------OVERALL METRICS--------")
          print("Overall Precision: ",np.mean(prec))
          print("Overall Recall: ",np.mean(rec))
          print("Overall F1-score: ",np.mean(f1))
          accuracy *= 100. / cnt
          print("Overall Accuracy: ",accuracy)


import numpy as np
import time
import datetime

def flat_accuracy(logits, labels,device):
  pred_flat = np.argmax(logits.cpu().detach().numpy(), axis=1).flatten()
  labels_flat = labels.cpu().detach().numpy().flatten()
  return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_preds(logits,device):
  preds = np.argmax(logits.cpu().detach().numpy(), axis=1).flatten()
  return preds.tolist()

def get_labels(labels,device):
  labels = labels.cpu().detach().numpy().flatten()
  return labels.tolist()

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
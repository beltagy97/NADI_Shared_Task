from transformers import get_linear_schedule_with_warmup,AdamW
from transformers import BertForSequenceClassification
from utils.helper_func import format_time

from utils.tokenizer import Tokenizer
from utils.data_loader import create_bert_dataloader

import pandas as pd
import time
import datetime
import random
import numpy as np
import torch
from run_model import run_model

# model = BertForSequenceClassification.from_pretrained(
#     "aubmindlab/bert-base-arabertv01", # Use the 12-layer BERT model, with an uncased vocab.
#     num_labels = 21, 
#     output_attentions = False, # Whether the model returns attentions weights.
#     output_hidden_states = False, # Whether the model returns all hidden-states.
# )
def train(train_loader,valid_loader,learning_rate=2e-5,eps=1e-8,model=None,device="cuda"):
  
  format_time(time.time()-time.time())
  model= BertForSequenceClassification.from_pretrained(
    "aubmindlab/bert-base-arabertv01", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 21, 
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    )
  
  model.to(device)
  
  optimizer = AdamW(model.parameters(),
                    lr = learning_rate, 
                    eps = eps)
  epochs = 4

  total_steps = len(train_loader) * epochs

  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)



  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # We'll store a number of quantities such as training and validation loss, 
  # validation accuracy, and timings.
  training_stats = []

  # Measure the total training time for the whole run.
  total_t0 = time.time()

  # For each epoch...
  for epoch_i in range(epochs):
      
      # ========================================
      #               Training
      # ========================================
      
      # Perform one full pass over the training set.

      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      # print('Training...')
      model.to(device)

      # Measure how long the training epoch takes.
      t0 = time.time()

      training_loss,training_acc=run_model(model,train_loader,True,optimizer,scheduler,device=device)
      print("training loss = "+str(training_loss))
      print("training acc = "+str(training_acc))
      print("-"*50)

      valid_loss,valid_acc = run_model(model,valid_loader,device=device)

      print("valid loss = "+str(valid_loss))
      print("valid acc = "+str(valid_acc))
      print("-"*50)

      # Calculate the average loss over all of the batches.
      # avg_train_loss = total_train_loss / len(train_dataloader)            
      
      # Measure how long this epoch took.
      training_time = format_time(time.time() - t0)

      # print("")
      # print("  Average training loss: {0:.2f}".format(avg_train_loss))
      # print("  Training epcoh took: {:}".format(training_time))

def main():
    
  training_data=pd.read_csv("data/preprocessed data/labeled training.csv")
  tweets = list(training_data["preprocessed tweet"])
  labels = list(training_data["label"])

  print("labels found")
  tokenizer = Tokenizer()
  input_ids,mask,labels=tokenizer.bert_tokenize_data(tweets,labels)

  tweets = list(zip(input_ids,mask))

  train_loader,valid_loader=create_bert_dataloader(tweets,labels)

  train(train_loader,valid_loader)





if __name__=="__main__":
  main()















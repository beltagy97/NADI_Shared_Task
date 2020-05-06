import time
from utils.helper_func import *
import torch

def run_model(model,data_loader,train=False,optimizer=None,scheduler=None,device="cuda"):

  if train:
    model.train()
  else :
    model.eval()


  # Reset the total loss for this epoch.
  total_loss = 0
  total_accuracy=0
  t0 = time.time()
  
  for step, batch in enumerate(data_loader):

          # Progress update every 40 batches.
          if step % 40 == 0 and not step == 0:
              # Calculate elapsed time in minutes.
              elapsed = format_time(time.time() - t0)
              
              # Report progress.
              print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(data_loader), elapsed))

          # Unpack this training batch from our dataloader. 
          #
          
          
          # `batch` contains three pytorch tensors:
          #   [0]: input ids 
          #   [1]: attention masks
          #   [2]: labels 
          input_ids = batch[0].to(device)
          input_mask = batch[1].to(device)
          labels = batch[2].to(device)

          model.zero_grad()        

          
          loss, logits = model(input_ids, 
                              token_type_ids=None, 
                              attention_mask=input_mask, 
                              labels=labels)

          
          total_loss += loss.item()
          total_accuracy += flat_accuracy(logits, labels,device)

          # Perform a backward pass to calculate the gradients.
          loss.backward()

          # Clip the norm of the gradients to 1.0.
          # This is to help prevent the "exploding gradients" problem.
          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

          if train:
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

  avg_train_loss = total_loss / len(data_loader)            
  avg_train_acc = total_accuracy / len(data_loader)
  return avg_train_loss,avg_train_acc
          

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def create_bert_dataloader(train_data,labels,train_size=0.95,batch_size = 32):

  input_ids, attention_masks = train_data
  
  dataset = TensorDataset(input_ids, attention_masks,labels)
  
  # Calculate the number of samples to include in each set.
  
  train_size = int(train_size * len(dataset))

  val_size = len(dataset) - train_size

  # Divide the dataset by randomly selecting samples.
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_dataloader = DataLoader(
              train_dataset,  # The training samples.
              sampler = RandomSampler(train_dataset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )

  validation_dataloader = DataLoader(
              val_dataset, # The validation samples.
              sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
              batch_size = batch_size # Evaluate with this batch size.
          )
  
  return train_dataloader,validation_dataloader

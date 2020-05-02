from transformers import BertForSequenceClassification

class Models():
  def __init__(self,name="bert_classifier",path=None,embedding=None):
    self.path=path
    self.name=name
  
  def get_model(self):
    if self.name =="bert_classifier":
      return self.get_bert_classifier()

    
    
  
  def get_bert_classifier(self):
    # if no saved model in path create a new one
    if self.path==None:
      return BertForSequenceClassification.from_pretrained(
          "aubmindlab/bert-base-arabertv01", # Use the 12-layer BERT model, with an uncased vocab.
          num_labels = 21, 
          output_attentions = False, # Whether the model returns attentions weights.
          output_hidden_states = False, # Whether the model returns all hidden-states.
          )



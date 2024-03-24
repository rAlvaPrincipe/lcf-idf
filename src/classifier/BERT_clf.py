import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import Linear, Module
import torch

class Layer(Module):
    def __init__(self, inp, out, act):
        super().__init__()
        self.fc = Linear(inp, out)
        self.act = act()

    def forward(self, x):
        return self.act(self.fc(x))

class BERT_Clf(LightningModule):
    def __init__(self, embedder, num_labels, ps, is_longformer):
        super().__init__()
        self.lr = ps["lr"]
        self.criterion = ps["criterion"]
        self.task = ps["task"]
        self.num_labels = num_labels
        self.embedder = embedder
        self.dropout = nn.Dropout(0.1)
        self.is_longformer = is_longformer
        
        if self.task == "multiclass" or self.task == "multilabel" or self.task == "multilabel-topone":
            self.linear = nn.Linear(self.embedder.config.hidden_size, num_labels)
        elif self.task == "binary":
            self.linear = nn.Linear(self.embedder.config.hidden_size, 1)
        
        
    def forward(self, input_ids, attention_mask):
        if self.is_longformer:
            global_attention_mask = torch.zeros_like(input_ids)  # creo un vettore di zeri delle dimensioni di input_ids
            global_attention_mask[:, 0] = 1  # solo il CLS token ha l'attenzione globale
            outputs = self.embedder(input_ids=input_ids, attention_mask=attention_mask,
                                                global_attention_mask=global_attention_mask)
        else:
            outputs = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        #pooled_output = outputs["last_hidden_state"][:,0]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits


    def loss_and_preds(self, logits, labels):
        if self.training:
            if self.task == "binary":
                loss = self.criterion(logits.view(-1), labels.type(torch.FloatTensor).cuda())
            elif self.task == "multiclass":
                loss = self.criterion(logits, labels)
            elif  self.task  == "multilabel-topone":
                loss = self.criterion(logits, labels.type(torch.FloatTensor).to(self.device))
            elif self.task == "multilabel":
                loss = self.criterion(logits, labels.type(torch.FloatTensor).to(self.device))
            return loss
        else:
            if self.task == "binary":
                loss = self.criterion(logits.view(-1), labels.type(torch.FloatTensor).cuda())
                preds =  torch.round(torch.sigmoid(logits)).squeeze().type(torch.IntTensor)
            elif self.task == "multiclass":
                loss = self.criterion(logits, labels)
                preds = torch.argmax(logits, axis=1)
            elif  self.task  == "multilabel-topone":
                loss = self.criterion(logits, labels.type(torch.FloatTensor).to(self.device))
                preds = torch.argmax(logits, axis=1)
            elif self.task == "multilabel":
                loss = self.criterion(logits, labels.type(torch.FloatTensor).to(self.device))
                preds = torch.round(torch.sigmoid(logits)).squeeze().type(torch.IntTensor)
            return loss, preds
        
            
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, attention_mask)
        loss = self.loss_and_preds(logits, labels)
        self.log('train_loss', float(f'{loss:.5f}'), on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, attention_mask)
        loss, preds = self.loss_and_preds(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss, 'val_preds': preds, 'val_labels': labels}
    
                  
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self.forward(input_ids, attention_mask)
        _, preds = self.loss_and_preds(logits, labels)
        return {'test_preds': preds, 'test_labels': labels}


    def validation_epoch_end(self, outputs):
        if not self.trainer.sanity_checking:
            tmp_pred, tmp_targets = list(), list()
            for output in outputs:
                tmp_pred.append(output["val_preds"].unsqueeze(0))  if output["val_preds"].dim() == 0 else tmp_pred.append(output["val_preds"])
                tmp_targets.append(output["val_labels"].unsqueeze(0))  if output["val_labels"].dim() == 0 else tmp_targets.append(output["val_labels"])
            self.val_predictions = torch.cat(tmp_pred)  
            self.val_targets = torch.cat(tmp_targets)
              
              
    def test_epoch_end(self, outputs):
        tmp_pred, tmp_targets = list(), list()
        for output in outputs:
            tmp_pred.append(output["test_preds"].unsqueeze(0))  if output["test_preds"].dim() == 0 else tmp_pred.append(output["test_preds"])
            tmp_targets.append(output["test_labels"].unsqueeze(0))  if output["test_labels"].dim() == 0 else tmp_targets.append(output["test_labels"])
        self.test_predictions = torch.cat(tmp_pred)  
        self.test_targets = torch.cat(tmp_targets)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

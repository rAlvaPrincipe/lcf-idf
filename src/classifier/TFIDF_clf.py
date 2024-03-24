import torch
from pytorch_lightning import LightningModule
from torch import nn

class TFIDF_Clf(LightningModule):
    def __init__(self, embedder, num_labels, ps):
        super().__init__()
        self.lr = ps["lr"]
        self.criterion = ps["criterion"]
        self.task = ps["task"]
        self.num_labels = num_labels
        self.embedder = embedder
        self.dropout = nn.Dropout(0.1)
        
        if self.task == "multiclass" or self.task == "multilabel" or self.task == "multilabel-topone":
            self.linear = nn.Linear(len(embedder.get_dictionary()), num_labels)
        elif self.task == "binary":
            self.linear = nn.Linear(len(embedder.get_dictionary()), 1)
        
        
    def forward(self, docs):
        x = self.embedder.embed(docs)
        x = x.to("cuda")
        x = self.dropout(x)
        x = self.linear(x)
        return x


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
        docs, labels = batch
        logits = self.forward(docs)
        loss = self.loss_and_preds(logits, labels)
        self.log('train_loss', float(f'{loss:.5f}'), on_step=False, on_epoch=True, prog_bar=True,  logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        docs, labels = batch
        logits = self.forward(docs)
        loss, preds = self.loss_and_preds(logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss, 'val_preds': preds, 'val_labels': labels}
    
                  
    def test_step(self, batch, batch_idx):
        docs, labels = batch
        logits = self.forward(docs)
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
    

import pytorch_lightning as pl
from src.classifier.metrics import print_metrics, show_confusion_matrix
import torchinfo
import os
import os.path
import shutil


class CustomCallback(pl.Callback):
    def __init__(self, dataset_name, model_name, task, id2cat, output_dir):
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.task = task
        self.id2cat = id2cat
        self.output_dir = output_dir
    
    
    def on_fit_start(self, trainer, pl_module):
        if os.path.exists("models/" + self.dataset_name + "/" + self.model_name + "/model.pt" ):
            os.remove("models/" + self.dataset_name + "/" + self.model_name + "/model.pt" )
            
        torchinfo.summary(pl_module, col_width=25, col_names=("num_params", "trainable"))
    
    
    def on_train_epoch_start(self, trainer, pl_module):
        print("---------------------------------------------------------------")
    
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            e = trainer.current_epoch
            label = "Epoch " + str(e + 1)
            output_f = self.output_dir + "/results/training_metrics.txt"
            print_metrics(pl_module.val_targets.cpu(), pl_module.val_predictions.cpu(), output_f, label, self.id2cat, self.task, trainer.callback_metrics["val_loss"])
            
            if e%10 == 0:
                output_cm = self.output_dir + "/results/cm/cm_" + str(e+1) + ".png"
                if self.task == "multiclass" or self.task == "binary":
                    show_confusion_matrix(pl_module.val_targets.cpu(), pl_module.val_predictions.cpu(), output_cm, self.id2cat)            

    
    
    def on_test_end(self, trainer, pl_module):
        output_f = "results/" + self.dataset_name + "/test_metrics.txt"
        output_f2 = self.output_dir + "/results/test_metrics.txt"
        print_metrics(pl_module.test_targets.cpu(), pl_module.test_predictions.cpu(), output_f, self.output_dir, self.id2cat, self.task)
        print_metrics(pl_module.test_targets.cpu(), pl_module.test_predictions.cpu(), output_f2, self.output_dir, self.id2cat, self.task)

    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.model_name == "BERT":
            pl_module.embedder.save_pretrained(self.output_dir +  "/" + "embedder")



class TestCallback(pl.Callback):
    def __init__(self, task, id2cat):
        super().__init__()
        self.task = task
        self.id2cat = id2cat

        
    def on_test_end(self, trainer, pl_module):
        print_metrics(pl_module.test_targets.cpu(), pl_module.test_predictions.cpu(), None, None, self.id2cat, self.task)

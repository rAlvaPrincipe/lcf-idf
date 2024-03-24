from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from pathlib import Path
import os
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def show_confusion_matrix(labels, predictions, output_f, id2cat):
    categories = list(id2cat.values())
    labels = [id2cat[l.item()] for l in labels]
    cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=categories)
    cmd_obj = ConfusionMatrixDisplay(cm, display_labels=categories)
    cmd_obj.plot()
    cmd_obj.ax_.set(
                    xlabel='Predicted Categories', 
                    ylabel='Actual Categories')
    plt.xticks(rotation=28, ha="right")

    Path(os.path.dirname(output_f)).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_f, bbox_inches="tight")


def print_metrics(labels, predictions, output_f, label, id2cat, label_type, val_loss=None):
    if label_type == "multilabel":
        report_human = classification_report(labels.tolist(), predictions.tolist(), target_names=list(id2cat.values()), digits=3)   
    elif label_type == "multilabel-topone":
        categories = categories = list(id2cat.values())
        labels_mod = []
        for lab, pred in zip(labels, predictions):
            if lab[pred.item()] == 1:
                labels_mod.append(pred.item())
            else:
                candidates = list(id2cat.keys())
                candidates.remove(pred.item())
                new_label = candidates[0]
                labels_mod.append(new_label)
        
        labels = labels_mod
        report_human = classification_report(labels, predictions, labels=categories, digits=3, output_dict=True)   
        report_human = str({"precision micro avg": round(report_human["micro avg"]["precision"], 3), "precision macro avg": round(report_human["macro avg"]["precision"], 3) })
            
    else:
        categories = list(id2cat.values())
        labels = [id2cat[l.item()] for l in labels]
        predictions = [id2cat[p.item()] for p in predictions]
        report_human = classification_report(labels, predictions, labels=categories, digits=3)

    print(report_human)
    if output_f:
        Path(os.path.dirname(output_f)).mkdir(parents=True, exist_ok=True)
        with open(output_f, "a") as f:
            f.write("\n" + label + " -------------------------------------------------------------\n")
            f.write(report_human)
            if val_loss:
                f.write(" ")
                f.write("Validation Loss: {:.6f}.. ".format(val_loss))


def error_analysis(labels, predictions, probs, ids, id2cat, output_f):
    labels = [id2cat[l.item()] for l in labels]
    errors = []
    for label, prediction, prob, id in zip(labels, predictions, probs, ids):
        if label != prediction:
            error = dict()
            error["doc_id"] = id
            error["gt"] = label
            error["prediction"] = prediction

            prob_dict = dict()
            for count, prob in enumerate(prob.tolist()):
                prob_dict[id2cat[count]] = round(prob, 4)
            error["probs"] = prob_dict

            errors.append(error)

    errors = pd.DataFrame(errors).sort_values(by=['gt'])
    Path(os.path.dirname(output_f)).mkdir(parents=True, exist_ok=True)
    errors.to_csv(output_f, index=False)


def print_message(output_f, message):
    with open(output_f, "a") as f:
        f.write("\n" +  message)
        
        
from src.classifier.BERT_clf import BERT_Clf
from src.classifier.TFIDF_clf import TFIDF_Clf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from src.classifier.dataloader import text_data_loader, bert_data_loader
from src.classifier.callbacks import CustomCallback, TestCallback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.classifier.TFIDF import TFIDF_Embedder
from src.translation_pipeline.translator import Translator
import torch
import os
from confs import parse, build_conf, load_conf

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def bert_NN(bert_ps, ps, x_train, y_train, x_val, y_val, x_test, y_test, num_labels, callbacks, output_dir):
    is_longformer = "longformer" in str(type(bert_ps["embedder"]))
    # dataLoader
    train_loader = bert_data_loader(bert_ps["tokenizer"], x_train, y_train, ps["batch"], bert_ps["ntokens"] )
    val_loader = bert_data_loader(bert_ps["tokenizer"], x_val, y_val, ps["batch"], bert_ps["ntokens"])
    test_loader = bert_data_loader(bert_ps["tokenizer"], x_test, y_test, ps["batch"], bert_ps["ntokens"])
    #training
    classifier = BERT_Clf(bert_ps["embedder"], num_labels, ps, is_longformer).to(ps["device"])
    trainer = Trainer(callbacks=callbacks, gpus=1, max_epochs=ps["epochs"], enable_model_summary=False)#, deterministic=True)
    trainer.fit(classifier, train_dataloaders= train_loader, val_dataloaders= val_loader)
    # testing
    classifier.load_state_dict(torch.load(output_dir + "/" +  "model.pt")["state_dict"])
    trainer.test(classifier, test_loader)
    
 
def tfidf_NN(embedder, ps, x_train, y_train, x_val, y_val, x_test, y_test, num_labels, callbacks, output_dir):
    train_loader = text_data_loader(x_train, y_train, ps["batch"])
    val_loader = text_data_loader(x_val, y_val, ps["batch"])
    test_loader = text_data_loader(x_test, y_test, ps["batch"])
    #training
    classifier = TFIDF_Clf(embedder, num_labels, ps).to(ps["device"])
    trainer = Trainer(callbacks=callbacks, gpus=1, max_epochs=ps["epochs"], enable_model_summary=False, deterministic=True)
    trainer.fit(classifier, train_dataloaders= train_loader, val_dataloaders= val_loader)
    # testing
    classifier.load_state_dict(torch.load(output_dir + "/" +  "model.pt")["state_dict"])
    trainer.test(classifier, test_loader)
       


def build_callbacks(test_mode, NN_ps, id2cat, model_name = None, dataset_name = None, output_dir = None):
    if test_mode:
        return [TestCallback(NN_ps["task"], id2cat)]
    else:
        custom_callback = CustomCallback(dataset_name, model_name, NN_ps["task"], id2cat, output_dir)
        earlystop_callback = EarlyStopping(monitor="val_loss", patience=NN_ps["patience"], mode="min", verbose=True)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min',save_top_k=1, dirpath=output_dir, filename="model")
        checkpoint_callback.FILE_EXTENSION = ".pt"
        return [custom_callback, earlystop_callback, checkpoint_callback]


def fit_tfidf_embedder(corpus, ps, output):
    tfidf_embedder = TFIDF_Embedder()
    tfidf_embedder.fit(corpus, "word", ps["ngrams"], ps["vocab_maxsize"])
    tfidf_embedder.save(output)
    return tfidf_embedder


def fit_translator(train_corpus, val_corpus, ps, output_dir):
    trans = Translator(ps["bert"]["tokenizer"], ps["bert"]["embedder"], output_dir, ps["reduction"],  ps["clustering"]["alg"], ps["clustering"]["n_clusters"])
    trans.fit(train_corpus, val_corpus)
    trans.save()
    return trans


###########################################


def inference_bert_NN(bert_ps, ps, x_test, y_test, num_labels, callbacks, output_dir):
    is_longformer = "longformer" in str(type(bert_ps["embedder"]))
    test_loader = bert_data_loader(bert_ps["tokenizer"], x_test, y_test, ps["batch"], bert_ps["ntokens"])
    classifier = BERT_Clf(bert_ps["embedder"], num_labels, ps, is_longformer).to(ps["device"])
    trainer = Trainer( callbacks=callbacks, gpus=1, max_epochs=ps["epochs"], enable_model_summary=False)#, deterministic=True)
    classifier.load_state_dict(torch.load(output_dir + "/" +  "model.pt")["state_dict"])
    trainer.test(classifier, test_loader)
    

def inference_tfidf_NN(embedder, ps, x_test, y_test, num_labels, callbacks, output_dir):
    test_loader = text_data_loader(x_test, y_test, ps["batch"])
    classifier = TFIDF_Clf(embedder, num_labels, ps).to(ps["device"])
    trainer = Trainer( callbacks=callbacks, gpus=1, max_epochs=ps["epochs"], enable_model_summary=False, deterministic=True)
    classifier.load_state_dict(torch.load(output_dir + "/" +  "model.pt")["state_dict"])
    trainer.test(classifier, test_loader)
       
       
def inference(model_dir):
    conf, model_type = load_conf(model_dir)
    seed = conf["seed"]
    seed_everything(seed, workers=True)
    
    _, id2cat, _, _, _, _, _, _, x_test, y_test, _ = conf["dataset"](conf["seed"])
    num_labels = len(id2cat.values())
    callbacks = build_callbacks(True, conf["nn"], id2cat)

    if model_type == "BERT":
        inference_bert_NN(conf["model"], conf["nn"], x_test, y_test, num_labels, callbacks, conf["output_dir"])
    elif model_type == "TFIDF":
        embedder = TFIDF_Embedder()
        embedder.load(conf["output_dir"] + "/embedder.pk")
        inference_tfidf_NN(embedder,  conf["nn"], x_test, y_test, num_labels, callbacks, conf["output_dir"])
    elif model_type == "CTFIDF":
        ps = conf["model"]
        translator = Translator(ps["bert"]["tokenizer"], ps["bert"]["embedder"], conf["output_dir"], ps["reduction"],  ps["clustering"]["alg"], ps["clustering"]["n_clusters"])
        translator.load(model_dir)
        x_test = translator.translate(x_test)
        embedder = TFIDF_Embedder()
        embedder.load(conf["output_dir"] + "/embedder/embedder.pk")
        inference_tfidf_NN(embedder,  conf["nn"], x_test, y_test, num_labels, callbacks, conf["output_dir"])


    
if __name__ == "__main__":
    #inference("lrec/models_1992_BCE/ecthr_small/bert-base-uncased")
    args = parse()
    seed_everything(args.seed, workers=True)
    
    conf, model_type = build_conf(args)
    dataset_name, id2cat, x_train, y_train, id_train, x_val, y_val, id_val, x_test, y_test, id_test = conf["dataset"](conf["seed"])
    print(len(x_train))
    print(len(x_val))
    print(len(x_test))
    input()
    num_labels = len(id2cat.values())

    #-------------------------- BERT -----------------------------------------
    if model_type == "BERT":
        callbacks = build_callbacks(False, conf["nn"], id2cat, model_type, dataset_name, conf["output_dir"])
        bert_NN(conf["model"], conf["nn"], x_train, y_train, x_val, y_val, x_test, y_test, num_labels, callbacks, conf["output_dir"])
    #-------------------------- TFIDF -----------------------------------------
    elif model_type == "TFIDF":
        embedder = fit_tfidf_embedder(x_train, conf["model"], conf["output_dir"] + "/embedder.pk")
        callbacks = build_callbacks(False, conf["nn"], id2cat, model_type, dataset_name, conf["output_dir"])
        tfidf_NN(embedder, conf["nn"], x_train, y_train, x_val, y_val, x_test, y_test, num_labels, callbacks,  conf["output_dir"])
    # ------------------------- C-TFIDF --------------------------------------
    elif model_type == "CTFIDF" :
        translator = fit_translator(x_train, x_val, conf["model"],  conf["output_dir"])
        x_train = translator.translate(x_train)
        x_val = translator.translate(x_val)
        x_test = translator.translate(x_test)

        embedder = fit_tfidf_embedder(x_train, conf["model"]["tfidf"],  conf["output_dir"] + "/embedder/embedder.pk")
        callbacks = build_callbacks(False, conf["nn"], id2cat, model_type, dataset_name, conf["output_dir"])
        tfidf_NN(embedder,  conf["nn"], x_train, y_train, x_val, y_val, x_test, y_test, num_labels, callbacks,  conf["output_dir"])


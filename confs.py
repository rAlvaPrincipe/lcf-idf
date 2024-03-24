import os
import argparse
from src.dataset.hyperpartisan import hyperpartisan
from src.dataset.newsgroups import newsgroups_small
from src.dataset.ecthr import ecthr_small
from src.dataset.eurlex import eurlex_small
from src.dataset.a_512 import a_512_small
from src.dataset.scotus import scotus_small
from transformers import LongformerModel

from transformers import AutoTokenizer, AutoModel
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from pathlib import Path
import json
import os
import hashlib
import time
from datetime import datetime
    
def bert_tok(version, lowercase):
    return AutoTokenizer.from_pretrained(version, do_lowercase=lowercase)


def bert_emb(version, train_embedder):
    if version == "allenai/longformer-base-4096":
        LongformerModel.from_pretrained("allenai/longformer-base-4096")
    if version == "bert-base-uncased" or version == "roberta-base" or version == "nlpaueb/legal-bert-base-uncased":
        bert = AutoModel.from_pretrained(version, output_hidden_states = True)
    else:
        bert =  AutoModel.from_pretrained(version, output_hidden_states = True, local_files_only=True)

    if train_embedder != None and not train_embedder:
        print("LOCKING..")
        for param in bert.parameters():
            param.requires_grad = False
    return bert

defaults = {
    "nn": {
        "batch": 8,
        "epochs": 3000,
        "lr": 3e-5,
        "patience": 5, 
        "device": "cuda",
    },
    "tfidf": {
        "ngrams": (1, 3),
        "vocab_maxsize": 100000
    },
    "ctfidf": {
        "reduction": [
            {
            "alg":"autoencoder", 
            "dim": 100
            }
        ], 
        "clustering": {
            "alg": "kmeans",
            "n_clusters": 200 
        },
        "tfidf": {
            "ngrams": (1, 3),
            "vocab_maxsize": 100000
        },  
    }  
}

def str2bool(v):
  if v == "true":
      return True
  elif v == "false":
      return False
  else:
      raise Exception("Unrecognized boolean value") 
  

def create_parser():
    parser = argparse.ArgumentParser(description="Text Classifier with TFIDF or BERT", allow_abbrev=False)
    
    # Add mutually exclusive group for choosing between TFIDF and BERT
    parser.add_argument('--tfidf', action='store_true', help='Use TFIDF as the model representation')
    parser.add_argument('--bert', action='store_true', help='Use BERT as the model representation')
    parser.add_argument('--ctfidf', action='store_true', help='Use TFIDF as the model representation')

    parser.add_argument('--seed', help='Name of the BERT tokenizer', type=int)
    parser.add_argument('--dataset', help='Name of the BERT tokenizer')
    parser.add_argument('--batch', help='Name of the BERT tokenizer')
    parser.add_argument('--epochs', help='Name of the BERT tokenizer')
    parser.add_argument('--task', help='Name of the BERT embedder')
    parser.add_argument('--criterion', help='Name of the BERT embedder')

    # Add TFIDF-specific arguments
    parser.add_argument('--ngrams', type=int, help='Number of n-grams for TFIDF')
    parser.add_argument('--num_max_terms', type=int, help='Number of max terms for TFIDF')

    # Add BERT-specific arguments
    parser.add_argument('--tokenizer', help='Name of the BERT tokenizer')
    parser.add_argument('--lowercase', help='Name of the BERT tokenizer')
    parser.add_argument('--embedder', help='Name of the BERT embedder')
    parser.add_argument('--finetune', help='Name of the BERT embedder')
    parser.add_argument('--ntokens', help='Name of the BERT embedder', type=int)
    
    
    # Add CTFIDF-specific arguments
    parser.add_argument('--dimreduction', help='Name of the BERT tokenizer')
    parser.add_argument('--nclusters', help='Name of the BERT tokenizer')
    parser.add_argument('--cluster_alg', help='Name of the BERT tokenizer')
    

    return parser


def parse():
    parser = create_parser()
    args = parser.parse_args()

    if not (args.seed and args.dataset and args.task and args.criterion):
        parser.error('seed, dataset, task and criterion are required')  
    else:
        if args.bert:
            if not (args.tokenizer and args.lowercase and args.embedder and args.ntokens and args.finetune):
                parser.error('When using a BERT-like model, --tokenizer, --lowercase, --embedder, --finetune and ntokens  are required.')  
        elif args.ctfidf:
            if not (args.tokenizer and args.lowercase and args.embedder and args.ntokens):
                parser.error('When using an C-TF-IIDF, --ngrams and --num_max_terms are required.')  
    if args.nclusters and args.cluster_alg and args.cluster_alg == "hdbscan" :
        parser.error('When using hdbscan you cannot provide a number of output clusters. Use Kmeans instead or remove nclusters parameter')  
    return args
    
    
def build_ouput_file_path(conf):
    output_dir = "models/"+ str(conf["seed"]) + "/" + conf["dataset"] + "/"
    if conf["model"]["type"] == "BERT":
        output_dir += "BERT_"
        if conf["model"]["embedder"].rfind('/') == -1:
            output_dir += conf["model"]["embedder"].replace("-","")
        else:
            output_dir += conf["model"]["embedder"][conf["model"]["embedder"].rfind('/')+1:-1].replace("-","")
        output_dir += "-lc" + str(conf["model"]["lowercase"])[0]
        output_dir += "-ft" + str(conf["model"]["finetune"])[0]
        output_dir += "-nt" + str(conf["model"]["ntokens"])
        
    elif conf["model"]["type"] == "TFIDF":
        output_dir += "TFIDF_"
        output_dir += "-ng" + str(conf["model"]["ngrams"])
        output_dir += "-max" + str(conf["model"]["vocab_maxsize"])
    
    elif conf["model"]["type"] == "CTFIDF":
        output_dir += "CTFIDF_"
        if "models/" in conf["model"]["bert"]["embedder"]:
            output_dir += "CUSTOM_"
            if "longformer" in conf["model"]["bert"]["embedder"]:
                output_dir += "longformer"
            elif "robertabase" in conf["model"]["bert"]["embedder"]:
                output_dir += "roberta"
            elif "legalbert" in conf["model"]["bert"]["embedder"]:
                output_dir += "legalbert"
            elif "bertbaseuncased" in conf["model"]["bert"]["embedder"]:
                output_dir += "bertbaseuncased"   
        else:  
            if conf["model"]["bert"]["embedder"].rfind('/') == -1:
                output_dir += conf["model"]["bert"]["embedder"].replace("-","")
            else:
                output_dir +=  conf["model"]["bert"]["embedder"][conf["model"]["bert"]["embedder"].rfind('/')+1:].replace("-","")
        output_dir += "-lc" + str(conf["model"]["bert"]["lowercase"])[0]
        output_dir += "-nt" + str(conf["model"]["bert"]["ntokens"])
        output_dir += "-" + conf["model"]["reduction"][0]["alg"]+ str(conf["model"]["reduction"][0]["dim"])
        if conf["model"]["clustering"]["alg"] == "kmeans":
            output_dir += "-" + conf["model"]["clustering"]["alg"] + str(conf["model"]["clustering"]["n_clusters"])
        elif conf["model"]["clustering"]["alg"] == "hdbscan":
            output_dir += "-" + conf["model"]["clustering"]["alg"]
        output_dir += "-ng" + str(conf["model"]["tfidf"]["ngrams"])
        output_dir += "-max" + str(conf["model"]["tfidf"]["vocab_maxsize"])
        
    output_dir += "__"
    output_dir += "b" + str(conf["nn"]["batch"])
    output_dir += "-lr" + str(conf["nn"]["lr"])
    output_dir += "-p" + str(conf["nn"]["patience"])
    output_dir += "-" + conf["nn"]["criterion"]
    
    conf["time"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    conf["version"] = hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]
    output_dir += "__V" + conf["version"]      
      
    return output_dir
          

def personalize(args, model_type):
    conf = {}
    conf["seed"] = args.seed 
    conf["dataset"] = args.dataset  
    conf["nn"] =  defaults["nn"] 
    conf["nn"]["task"] = args.task
    conf["nn"]["criterion"] = args.criterion
    if args.batch:
        conf["nn"]["batch"] = int(args.batch)
    if args.epochs:
        conf["nn"]["epochs"] = int(args.epochs)

    if model_type == "BERT":
        conf["model"] = {"type": model_type, "tokenizer": args.tokenizer, "lowercase": str2bool(args.lowercase), "embedder": args.embedder, "finetune": str2bool(args.finetune), "ntokens": args.ntokens}
    elif model_type == "TFIDF":
        conf["model"] = {"type": "TFIDF"}
        #set defaults
        conf["model"]["ngrams"] = defaults["tfidf"]["ngrams"]
        conf["model"]["vocab_maxsize"] = defaults["tfidf"]["vocab_maxsize"]
        #update defaults
        if args.num_max_terms:
            conf["model"]["vocab_maxsize"] = int(args.num_max_terms)
        conf["model"]["type"] = model_type
    elif model_type == "CTFIDF":  
        conf["model"] = {"type": "CTFIDF"}
        conf["model"]["bert"] = {"tokenizer": args.tokenizer, "lowercase": str2bool(args.lowercase), "embedder": args.embedder, "ntokens": args.ntokens}
        #set defaults
        conf["model"]["tfidf"] =  defaults["ctfidf"]["tfidf"]
        conf["model"]["reduction"] = defaults["ctfidf"]["reduction"]
        conf["model"]["clustering"] = defaults["ctfidf"]["clustering"]
        #update defaults
        if args.dimreduction:
            conf["model"]["reduction"][0]["dim"] = int(args.dimreduction)
        if args.cluster_alg:
            if args.cluster_alg == "hdbscan":
                conf["model"]["clustering"] = {}
                conf["model"]["clustering"]["alg"] = "hdbscan"
        if args.nclusters:
            conf["model"]["clustering"]["n_clusters"] = int(args.nclusters)
        if args.num_max_terms:
            conf["model"]["tfidf"]["vocab_maxsize"] = int(args.num_max_terms)
            
    
    conf["output_dir"] = build_ouput_file_path(conf)       
    return conf


def save(conf):
    f_out = conf["output_dir"] + "/conf.json"
    Path(os.path.dirname(f_out)).mkdir(parents=True, exist_ok=True)
    with open(f_out, 'w') as fp:
        json.dump(conf, fp, indent=4)
        
 
def objectify(conf, model_type):
    if conf["dataset"] == "hyperpartisan":
        conf["dataset"] = hyperpartisan
    elif conf["dataset"] == "ecthr_small":
        conf["dataset"] = ecthr_small
    if conf["dataset"] == "eurlex_small":
        conf["dataset"] = eurlex_small
    if conf["dataset"] == "a_512_small":
        conf["dataset"] = a_512_small
    if conf["dataset"] == "scotus_small":
        conf["dataset"] = scotus_small
    if conf["dataset"] == "newsgroups_small":
        conf["dataset"] = newsgroups_small
        
    if conf["nn"]["criterion"] == "bce":
        conf["nn"]["criterion"] = BCEWithLogitsLoss()
    elif conf["nn"]["criterion"] == "ce":
        conf["nn"]["criterion"] = CrossEntropyLoss()
        
    if model_type == "BERT":
        conf["model"]["tokenizer"] = bert_tok(conf["model"]["tokenizer"], conf["model"]["lowercase"])
        conf["model"]["embedder"] = bert_emb(conf["model"]["embedder"], conf["model"]["finetune"])

    elif model_type == "CTFIDF":
        conf["model"]["bert"]["tokenizer"] = bert_tok(conf["model"]["bert"]["tokenizer"], conf["model"]["bert"]["lowercase"])
        conf["model"]["bert"]["embedder"] = bert_emb(conf["model"]["bert"]["embedder"], False) # in C-TFIDF you cannot train BERT
        if conf["model"]["clustering"]["alg"] == "hdbscan":
            conf["model"]["clustering"]["n_clusters"] = None

    return conf

    
def build_conf(args):
    if args.bert:
        model_type = "BERT"
    elif args.tfidf:
        model_type = "TFIDF"
    elif args.ctfidf:
        model_type = "CTFIDF"
        
    conf = personalize(args, model_type)
    save(conf)
    conf = objectify(conf,model_type)
    return conf, model_type
    

def load_conf(output_dir):
    f = output_dir + "/conf.json"
    with open(f, "r") as file:
        conf = json.load(file)
    
    if "version" not in conf.keys():
        conf["nn"] = conf["NN"]
        del conf['NN'] 
        if conf["nn"]["criterion"] == "CrossEntropyLoss":
            conf["nn"]["criterion"] = "ce"
        elif conf["nn"]["criterion"] == "BCEWithLogitsLoss":
            conf["nn"]["criterion"] = "bce"
        conf["dataset"]   = conf["dataset"]["dataset"]
        conf["output_dir"] = output_dir
        conf["seed"] = conf["custom_Seed"]
        
        if "clustering" in conf["model"]:
            conf["model"]["type"] = "CTFIDF"
            conf["model"]["tfidf"]["ngrams"] = (conf["model"]["tfidf"]["ngrams"][0], conf["model"]["tfidf"]["ngrams"][1])
            conf["model"]["bert"]["ntokens"] = 512
        elif "ngrams" in conf["model"]:
            conf["model"]["type"] = "TFIDF"
            conf["model"]["ngrams"] = (conf["model"]["ngrams"][0], conf["model"]["ngrams"][1])
        elif "embedder" in conf["model"]:
            conf["model"]["type"] = "BERT"
            conf["model"]["finetune"] = conf["model"]["fine_tune"]
            conf["model"]["ntokens"] = 512
    model_type = conf["model"]["type"]
    conf = objectify(conf, model_type)
    return conf, model_type
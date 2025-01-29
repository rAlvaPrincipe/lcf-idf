# An LCF-IDF Document Representation Model Applied to Long Document Classification

Questo repository implementa il modello LCF-IDF (Latent Concept Frequency - Inverse Document Frequency).
Questo modello è pensato per generare una rappresentazione vettoriale di documenti lunghi.
Da qui è possibile anche eseguire diverse baseline TF-IDF e BERT-based.
Per maggiori dettali fare riferimento a: https://aclanthology.org/2024.lrec-main.101/


<img src="images/concept_discovery_and_translation.png" alt="Caption for the image" width="1000">



<img src="images/lcfidf.png" alt="Caption for the image" width="1000">


## Usage:

### Parameters:

Scelta famiglia modello:

- tfidf: è un flag. Se usato usa tfidf
- bert: è un flag. Se usato usa un modello BERT-like
- ctfidf: è un flag. Se usato usa LCF-IDF

Training: 

- seed: seed per riproduciblità
- dataset: dataset su cui eseguire l'evaluation
- batch: grandezza del batach durante il training
- epochs: #epoche di training
- task: può essere binary, multiclass o multilabel. Da usare correttamente in base al dataset
- criterion: "bce" (binary cross-entropy) o "ce" (cross-entropy). Da associare correttamente al task

Specifici per TF-IDF:

- ngrams: #ngrams per TF-IDF
- num_max_terms: grandezza massima covabolario per TF-IDF

Specifici per modelli BERT-like:

- tokenizer: huggingface tokenizer
- lowercase: se applicare un lowercasing prima della tokenizzazione
- embedder: huggingface embedder
- finetune: allenare i pesi del modello oppure solo il classificatore
- ntokens: #tokens della finestra di input

Specifici per LCF-IDF:

- dimreduction: che algoritmo usare nella fase di riduzione della dimensionalità
- nclusters: #clusters nella fase di concept discovering
- cluster_alg: che algoritmo usare nel clustering



### Models:

- allenai/longformer-base-4096
- bert-base-uncased
- roberta-base
- nlpaueb/legal-bert-base-uncased
- dbmdz/bert-base-italian-uncased
- dbmdz/bert-base-italian-xxl-uncased
- dlicari/Italian-Legal-BERT-SC
- LCF-IDF usando come embedder backend i modelli sopracitati
- TF-IDF

### Datasets:

- hyperpartisan
- newsgroups_small
- ecthr_small
- eurlex_small
- a_512_small
- scotus_small
- protos2
- protos9
- protos10
- protos19
- protos20
- protos_all


### Run (examples):
BERT-like
```
$ venv/bin/python main.py  --seed 2003 --dataset protos_all --bert --tokenizer dbmdz/bert-base-italian-xxl-uncased --lowercase true --embedder dbmdz/bert-base-italian-xxl-uncased  --finetune true  --ntokens 512 --task multiclass --criterion ce
```

TF-IDF:
```
$ venv/bin/python main.py  --seed 12345 --dataset protos_2 --tfidf --lowercase true  --task binary --criterion bce
```

LCF-IDF:
```
$ venv/bin/python tmp_main.py  --seed 1992 --dataset eurlex_small --ctfidf --tokenizer allenai/longformer-base-4096 --lowercase true --embedder allenai/longformer-base-4096  --ntokens 4096 --task multilabel-topone --criterion ce
```

LCF-IDF Custom:
```
$ venv/bin/python main.py  --seed 12345 --dataset protos_2 --ctfidf --tokenizer dbmdz/bert-base-italian-xxl-uncased --lowercase true --embedder models/12345/protos_2/BERT_bertbaseitalianxxluncase-lcT-ftT-nt512__b8-lr3e-05-p5-bce__Vee13/embedder  --ntokens 512 --task binary --criterion bce
```

### Results:


<img src="images/open_datasets.png" alt="Caption for the image" width="1000">

*Here, the mu symbol stands for "small" in the Datasets section*


<img src="images/datasinc_datasets.png" alt="Caption for the image" width="1000">

*Here, DatasincX means  protosX where X can be 2,9,10,19,20,all*

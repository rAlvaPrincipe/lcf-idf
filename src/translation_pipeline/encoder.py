from transformers import BertTokenizer, BertModel
import torch

class Encoder():

    def __init__(self, tokenizer, embedder, device):
        self.device = device
        self.tokenizer = tokenizer
        self.embedder = embedder.to(self.device)


    def doc_segmenter(self, doc, chunk_size=510):
        '''
        input: doc (stringa), chunk_size (intero, considera che ogni chunk verrà negli step dopo arricchito con CLS e SEP, per questo il default è 510)
        output: lista di chunks (lista di liste). Ogni chunk è una sentence che BERT può digerire per intero.
        es: "ciao mi chiamo Renzo e sono un dottorato" chunk_size= 5   --> [["ciao", "mi", "chiamo", "Renzo", "e"], ["sono", "un", dott", "##or", "##ato"]]
        TODO questo processo deve anche: (1) segnarsi dove sono le subword (2) non avere delle subwods a cavallo di chunk diversi (3) gestire gli unknown
        '''
        tokens = self.tokenizer.tokenize(doc)
        chunks = [tokens[i:i+chunk_size] for i in range(0,len(tokens), chunk_size)]
        return chunks


    def prepare_segment_for_model(self, segment):
        '''
        input:  [["ciao", "mi", "chiamo", "Renzo", "e"], ["sono", "un", dott", "##or", "##ato"]]
        output: [[101, 32, 33, 24, 11, 7, 102], [101, 345, 22, 12, 77, 3, 102]]
        '''
        encoded = self.tokenizer.convert_tokens_to_ids(segment)
        encoded = self.tokenizer.prepare_for_model(encoded)
        ids_tensor = torch.tensor([encoded["input_ids"]]).to(self.device)
        masks_tensor = torch.tensor([encoded["attention_mask"]]).to(self.device)
        return ids_tensor, masks_tensor


    def get_model_hidden_states(self, ids_tensor, masks_tensor):
        ''' 
        input: lista ordinata di tokens che rappresenta una sentence 
        output: tensore [# tokens, # layers, # features] 
        '''
        self.embedder.eval()
        with torch.no_grad():
            outputs = self.embedder(ids_tensor, masks_tensor)      ##!!!!!!!!!!!1
            hidden_states = outputs['hidden_states']            #  --> tuple (13 elements, where each: [# batches, # tokens, # features])   (the same of outputs[2])
            hidden_states = torch.stack(hidden_states, dim=0)   #  -->  [# layers, # batches, # tokens, # features]                                                      ##!!!!!!!!!!!1
            hidden_states = torch.squeeze(hidden_states, dim=1) #  -->  [# layers, # tokens, # features]
            hidden_states = hidden_states.permute(1,0,2)        #  -->  [# tokens, # layers, # features]
            return hidden_states


    def avg_last_n_layers_embeddings(self, hidden_states, n_layers):
        '''
        word vectors by summing together the last four layers.
        input: all the hidden layers ([# tokens, # layers, # features]), n_layers (the number of layers to keep for the average)
        output: un tensore nX769 dove n = # tokens
        '''
        word_embeddings = []
        for token in hidden_states:
            embedding = torch.sum(token[-n_layers:], dim=0)
            embedding = embedding.to("cpu")                   # necessary because you need to store a huge ammount of vectors to perform clustering
            word_embeddings.append(embedding)
        #print ('Shape is: %d x %d' % (len(word_embeddings), len(word_embeddings[0])))
        return word_embeddings


    def corpus2wordembeddings(self, corpus):
        '''
        input: the corpus (a list of strings)
        output: a list of contextual word embeddings (1word --> 1 embedding)
        '''
        out = list()
        for doc in corpus:
            doc_segments = self.doc_segmenter(doc, 510)            # segmento il doc in chunks gestibili da BERT
            for segment in doc_segments:
                ids_tensor, masks_tensor = self.prepare_segment_for_model(segment)
                hidden_states = self.get_model_hidden_states(ids_tensor, masks_tensor)
                embeddings = self.avg_last_n_layers_embeddings(hidden_states, 4)
                out.extend(embeddings)

        return torch.stack(out)  ##!!!!!!!!!!!1



    def get_dictionary(self):
        4
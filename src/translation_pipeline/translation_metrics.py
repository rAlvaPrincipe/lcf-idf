from collections import Counter
from collections import OrderedDict
import matplotlib.pyplot as plt
from statistics import  mean
import pprint

def concepts_distribution(x, y, prefix_name):
    if type(y[0]) == list:
        temp_x = []
        temp_y = []
        for text, labels in zip(x, y):
            for label in labels: 
                temp_x.append(text)
                temp_y.append(label)
        x = temp_x
        y = temp_y
    
    avg_concepts_per_doc(x,y)
    avg_common_concepts(x,y)
    
    labels = set(y)
    corpus = dict()
    for label in labels:
        corpus[label] = ""
    
    for text, label in zip(x, y):
        corpus[label] += " " + text

    percs = dict()
    maximum_perc = 0
    for label in labels:        
        l = len(corpus[label].split())
        c = Counter(corpus[label].split())
        perc = dict([(i, c[i] / l * 100.0) for i in c])
        perc= dict(OrderedDict(sorted(perc.items(), key=lambda t: int(t[0]))))
        
        percs[label] = perc
        for concept in perc.keys():
            if perc[concept] > maximum_perc:
                maximum_perc = perc[concept]
                
    for label in percs.keys():    
        plot( percs[label], str(label), prefix_name, maximum_perc)


# quanti concetti utilizzano in media i documenti nella classe X
def avg_concepts_per_doc(x, y):
    labels = set(y)
    counts = dict()
    
    for label in labels:
        counts[label] = list()
    
    for text, label in zip(x, y):
        counts[label].append(len(set(text.split())))
        
    for label in labels:
        counts[label] = mean(counts[label])
    
    pprint.pprint(counts)
    
    
# Considerando  x come un unico documento quanti concetti ha in comune in media con le altre classi?
def avg_common_concepts(x, y):
    corpus = dict()
    intersections = dict()
    labels = set(y)
    for label in labels:
        corpus[label] = ""
        intersections[label] = []
        
    for text, label in zip(x, y):
        corpus[label] += " " + text
        
    for label in corpus.keys():   
        corpus[label] = set(corpus[label].split())
    
    for label_i in corpus.keys():
        corpus[label]
        for label_j in corpus.keys():
            if label_i != label_j:
                intersections[label_i].append(len(corpus[label_i].intersection(corpus[label_j])))

    for label in intersections.keys():
        intersections[label] = mean(intersections[label] )
        
    pprint.pprint(intersections)
        


def plot(data_dict, label, prefix_name, y_lim):
    plt.re
    # Extract keys and values
    keys = list(map(int, data_dict.keys()))
    values = list(data_dict.values())

    # Create histogram
    plt.bar(keys, values)
    plt.title('Histogram of Values')
    plt.xlabel('Keys')
    plt.ylabel('Values')
    plt.xticks(keys)
    plt.ylim(0, y_lim) 
    
    # Save the plot to disk
    plt.savefig("plots/" + prefix_name +"_" + label +'.png')
    plt.clf()

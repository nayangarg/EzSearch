# 1.
import re
import os
import nltk
import string
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
from read_write_create import *
#import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import sys
import csv
import unicodedata

reload(sys)
sys.setdefaultencoding('utf8')


def convert_text_to_sentences(text):
    return nltk.sent_tokenize(text)


def select_adj_noun(POS_tags):
    adj_ex = re.compile("JJ.?")
    noun_ex = re.compile("NN.?.?")

    selected_words = []
    for word, tag in POS_tags:
        if adj_ex.match(tag) or noun_ex.match(tag):
            selected_words.append(word)

    return selected_words


def create_graph_with_adjmat(adjacency_matrix, labels):
    num_of_nodes = len(adjacency_matrix)

    rows, cols = np.where(adjacency_matrix > 0)
    edges = zip(rows.tolist(), cols.tolist())

    nodes_with_labels = []
    for i in range(num_of_nodes):
        tup = (i, dict(labels=labels[i]))
        nodes_with_labels.append(tup)

    wted_edges = []
    for edge in edges:
        wt = adjacency_matrix[edge[0], edge[1]]
        tup = (edge[0], edge[1], wt)
        wted_edges.append(tup)

    gr = nx.Graph()
    gr.add_nodes_from(nodes_with_labels)
    gr.add_weighted_edges_from(wted_edges)

    ## to print graph
    # nx.draw(gr, node_size=500, labels=labels, with_labels=True)
    # plt.show()

    return gr


def create_graph_with_edgelist():
    return 1


global path, data_path, position_path

# cwd = os.getcwd()
# path = '/home/nayan/coding/Major/EasySearch/venv'
li = sys.argv[1:]

if '/' in li[0]:
    path = li[0]
    data_path = path + '/data/'
    
    position_path = path + "/positions/"
    create_folder(position_path)
    create_folder(path + "graphs")
    create_folder(path + "edgelists")
    create_folder(path + "wordList")
    li = os.listdir(data_path)

ps = PorterStemmer()
# lemmatizer = WordNetLemmatizer()
stopwords = read_list_from_file(path, "stopwords.txt")
roman_num_ex = re.compile("\\b[i|v|x|l|c|d|m]{1,3}\\b")
numbers_ex = re.compile("[0-9]+(-[0-9]+)?")
punctuation_ex = re.compile("[^a-z ]")
newline_ex = re.compile("\n")
hyphen = re.compile("-")

# print("Create-graph-sCake")
# li = sys.argv[1:]
# li = ['566390.txt']
for every_file in li:

    file_name = every_file[:-4]
    # print(every_file)

    #Pre-Processing Text

    text = read_text_from_file(data_path, every_file)
    sen = convert_text_to_sentences(text)
    text = read_text_from_file(data_path, every_file)
    text = re.sub(newline_ex, ' ', text)
    text = re.sub(hyphen, ' ', text)
    text = unicode(text, errors='replace')

    
    text = text.strip()
    text = text.lower()
    text = re.sub(numbers_ex, '', text)
    text = re.sub(punctuation_ex, '', text)
    text = re.sub(roman_num_ex, '', text)

    words = nltk.word_tokenize(text)
    # print(words[0])
    words = [i for i in words if i not in stopwords]
    words = [ps.stem(i) for i in words]

    l = len(words)
    i = 0
    # print(words[0])

    while i < l:
        #print(i,l)
        if len(words[i]) < 3 :
            words.pop(i)
            i -= 1
            l -= 1
        i += 1
    
    # Finding Bi-tri

    bigrams = nltk.collocations.BigramAssocMeasures()
    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(words)

    bigramFinder.apply_freq_filter(4)
    bi = list(bigramFinder.score_ngrams(bigrams.pmi))
    

    trigrams = nltk.collocations.TrigramAssocMeasures()
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(words)
    trigramFinder.apply_freq_filter(4)

    tri = list(trigramFinder.score_ngrams(trigrams.pmi))

    text = read_text_from_file(data_path, every_file)
    text = re.sub(newline_ex, ' ', text)
    text = re.sub(hyphen, ' ', text)
    text = unicode(text, errors='replace')

    
    # sen = convert_text_to_sentences(text)
    # print(sen[0])

    #Title Abstract Body Reference

    ab_sen=0
    t_sen=0
    body_sen=0
    r_sen=0
    flag=0
    l = len(sen)
    for i in range(l):
        if(sen[i][0:3]=='--T'):
            while(sen[i][0:3]!='--A'):
                t_sen=t_sen+1
                i=i+1
            i=i-1
        if(sen[i][0:3]=='--A'):
            while(sen[i][0:3]!='--B'):
                ab_sen=ab_sen+1
                i=i+1
            i=i-1
        if(sen[i][0:3]=='--B'):
            while(sen[i][0:3]!='--R'):
                body_sen=body_sen+1
                i=i+1
            i = i-1
        if(sen[i][0:3]=='--R'):
            break
    r_sen=len(sen)-t_sen-ab_sen-body_sen

    # print(t_sen,ab_sen,body_sen, r_sen, len(sen))

    # if file_name == 'survey paper on keyword extraction':
    #     print(sen[0])
    # print(type(sen))

    # Creating WordList
    sen = convert_text_to_sentences(text)
    # print(len(sen))

    wrtr = csv.writer(open(path + "/wordList/" + every_file[:-4] + ".csv", "w"))

    senWord = []
    wL = []

    for s in sen:

        new_sen = s.strip()
        new_sen = new_sen.lower()
        new_sen = re.sub(numbers_ex, '', new_sen)
        new_sen = re.sub(punctuation_ex, '', new_sen)
        new_sen = re.sub(roman_num_ex, '', new_sen)
        # print(new_sen)
        words = nltk.word_tokenize(new_sen)
        # print(words)

        words = [i for i in words if i not in stopwords]
        # print(words)
        words = [ps.stem(w) for w in words]
        # words = [lemmatizer.lemmatize(i) for i in words]

        w = 0
        while w < len(words) - 2:
            # print('bahar', w, len(words))
            for b in bi:
                # print('andar', w, len(words))
                if words[w] == b[0][0] and words[w + 1] == b[0][1]:

                    f = 0
                    for t in tri:
                        if b[0][0] == t[0][0] and b[0][1] == t[0][1]:
                            f = 1
                            words[w] = words[w] + '-' + words[w + 1] + '-' + words[w + 2]
                            # words[w] = words[w].encode('utf-8')
                            words.pop(w + 1)
                            words.pop(w + 1)
                            break

                    if f is 0:
                        words[w] = words[w] + '-' + words[w + 1]
                        # words[w] = words[w].encode('utf-8')
                        words.pop(w + 1)

                    break
            w += 1

       	
        wrtr.writerow(words)
        senWord.append(words)

        for i in words:
        	wL.append(i)

    #Creating Positions

    # f_name=every_file[:-4]+".csv"
    # t = read_list_from_file(cwd+"/wordList",f_name)
    # print(f_name)
    t_len=0
    ab_len=0
    body_len=0
    r_len=0

    l = len(senWord)

    # print(t[:10])

    count=0
    for i in range(l):
        count+=1
        if(count<=t_sen):
            t_len += len(senWord[i])

        elif(count>t_sen and count<=(ab_sen+t_sen)):
            ab_len += len(senWord[i])

        elif(count>(ab_sen+t_sen) and count<=(ab_sen+t_sen+body_sen)):
            body_len += len(senWord[i])

        else:
            r_len += len(senWord[i])  
            

    selected_words = list(set(wL))

    N = len(wL) + 1
    posi = list()
    # ori= list()
    t = list()
    tf = list()

    for w in selected_words:

        if len(w) > 2:
            posw = [i for i, word in enumerate(wL) if w == word]

            l1 = len(posw)
            # poso = [i for i, word in enumerate(wL) if w == word]   #me
            for x in range(l1):
                if(posw[x]>0 and posw[x]<t_len):
                    posw[x]=posw[x]/7
                if(posw[x]>t_len and posw[x]<ab_len):
                    posw[x]=posw[x]/6
                if(posw[x]>ab_len and posw[x]<body_len):
                    posw[x]=posw[x]/1
                else:
                    posw[x]=posw[x]/1.5
            w_freq = len(posw) + 1
            posw.append(N)
            t.append(w)
            tf.append(w_freq)
            posi.append(posw)
            # ori.append(poso)

    data = dict()
    data["words"] = t
    data["tf"] = tf
    data["positions"] = posi

    df_pos = pd.DataFrame(data=data)
    # print(df_pos.head())

    df_pos.to_pickle(path + "/positions/" + every_file[:-4] + ".pkl") #me
        # print(words)


    ## creating corpus s.t. two sentences are considered one document.

    doc = list()
    if len(senWord) < 2:
        doc.append(senWord[0])
    else:
        for i in range(len(senWord) - 1):
            two_sen = list()

            for j in senWord[i]:
                two_sen.append(j)

            for j in senWord[i+1]:
                two_sen.append(j)

            doc.append(two_sen)
	    	
    
    corpus = []

    for d in doc:
    	# print(d)
    	# x = d[0]
    	# print(x)
    	# for i in d:
    	x = ' '.join(d)
    	# print(x)
    	corpus.append(x)
    # print(corpus[0])

    words = df_pos["words"]
    selected_words = sorted(list(set(words)))
    # print(doc[0])
    # print(corpus[0])

    ##create document-term matrix
    vectorizer = CountVectorizer(binary=True, token_pattern='[a-zA-Z-]+')
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names()
    # print(features)
    # print(len(features))

    # only selected_words used
    ind_dict = dict((k, i) for i, k in enumerate(features))
    ins = list((set(ind_dict)) & (set(selected_words)))
    indices = [ind_dict[x] for x in ins]

    dtm = (X[:, indices]).todense()

    ##create term-term matrix
    ttm = np.transpose(dtm) * dtm
    np.fill_diagonal(ttm, 0)

    df = pd.DataFrame(data=ttm)
    df.columns = ins

    # print(df)
    df.to_pickle(path + "/graphs/" + every_file[:-4] + ".pkl")

    ##create graphs from ttm
    labels = dict((i, k) for i, k in enumerate(ins))

    G = create_graph_with_adjmat(ttm, labels)
    # edge_list = nx.generate_edgelist(G)
    edge_list = G.edges().data()
    # print(G.number_of_nodes())
    # print(labels)

    g = nx.Graph()

    edgelist = []
    for line in edge_list:
        tup = (labels[line[0]], labels[line[1]], line[2]['weight'])
        # g.add_edge()
        edgelist.append(tup)

    df = pd.DataFrame(edgelist)
    df.to_csv(path + "/edgelists/" + every_file[:-4] + ".csv", sep='\t', header=False, index=False)

    # pos = nx.spring_layout(G)  # positions for all nodes
    #
    # elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]
    #
    # # nodes
    # nx.draw_networkx_nodes(G, pos, node_size=700, node_color = 'yellow')
    #
    # # edges
    # nx.draw_networkx_edges(G, pos, edgelist=elarge,
    #                        width=1)
    # nx.draw_networkx_edges(G, pos, edgelist=esmall,
    #                        width=1, alpha=0.5, edge_color='b', style='dashed')
    #
    # # labels
    # nx.draw_networkx_labels(G, pos, labels, font_size=15, font_family='sans-serif', font_color = 'blue')
    #
    # plt.axis('off')
    # plt.show()

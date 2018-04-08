import io
import os
import sys
import csv
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json 

# Import nltk package 
#   PennTreeBank word tokenizer 
#   English language stopwords
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from collections import OrderedDict, Counter

import docx

cl = 10

#Download MLTK English tokenizer/stopwords 
#nltk.download('punkt')
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation)
#read in dataset
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
    	str_para = para.text.translate(translator)
    	fullText.append(str_para.lower())
    return fullText

data = getText('CogsNotes.docx')

df_results = pd.DataFrame(data=data)
df_results.columns = ['results']
df_results['results'].replace('', np.nan, inplace=True)
df_results['results'].replace('\n', np.nan, inplace=True)
df_results.dropna(subset=['results'], inplace=True)
df_results.reset_index(drop=True, inplace=True)
print(df_results.head())


#create TfidfVectorizer object
tfidf = TfidfVectorizer(sublinear_tf=True, analyzer='word', 
						max_features=100, tokenizer=word_tokenize, max_df=0.8)


#print("Transforming dataset")


tf = tfidf.fit_transform(df_results["results"])
tf = tf.toarray() #transformed vectors into a matrix


#Check if dataset shape lines up: for this case should be 10000 and 1000
#print(tf.shape)


#print("plotting datapoints on grid")
#Plot data on a 2 dimensional grid just for visualization purposes
pca = PCA(n_components=2).fit(tf)
data2D = pca.transform(tf) #data2D = number of posts * 2
#plt.show()  

#print("plotting k-means analysis")
#use K-means on data and plot means on PCA graph 
kmeans = MiniBatchKMeans(n_clusters=cl, random_state=500).fit(tf) #default threshold
centers2D = pca.transform(kmeans.cluster_centers_)

#plt.show()              


#convert clusters back into text
clusters = kmeans.cluster_centers_
#print(clusters)
questions_clusters = tfidf.inverse_transform(X=clusters)

#print(tfidf.get_feature_names())
tfidf_dict = tfidf.vocabulary_

plt.scatter(data2D[:, 0], data2D[:, 1], c=kmeans.labels_)
plt.scatter(centers2D[:,0], centers2D[:,1], 
	marker='x', s=200, linewidths=3, c='r')

#print("all plotted")

plt.show()

#sort sentences based on label/tfidf
df_results["label"] = pd.Series(kmeans.labels_, index = df_results.index)
print(df_results.head())

def getStrings(input_df):
	output = ""
	for string in input_df:
		output = output + " " +string
	return output.lower()

total = getStrings(df_results["results"])
words = word_tokenize(total)
stopWords = set(stopwords.words("english"))

freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

sentences = []
def getStrings(input_df):
	list_of_text = []
	for string in input_df:
		list_of_text.append(string)
	return list_of_text

sentences = df_results["results"]
sentenceValue = np.zeros(len(sentences))
#print(len(sentenceValue))

index = 0
for i in sentences:
	words_in_sent = word_tokenize(i)
	for j in words_in_sent:
		j = j.lower()
		if j in stopWords:
			pass
		else:
			sentenceValue[index] += 1
	index += 1

sumValues = 0
for sentence in sentenceValue:
    sumValues += sentence

# Average value of a sentence from original text
average = int(sumValues/ len(sentenceValue))

labels = df_results['label']
current_label = labels[0]
summary = ''
summary_ind = []
index = 0
for sentence in sentences:
    if sentenceValue[index] > (1.2 * average):
    	words_in_sent = word_tokenize(sentence)
    	filtered_sentence = [w for w in words_in_sent if not w in stop_words]
    	str1 = ' '.join(filtered_sentence)
    	summary +=  "\n" + str1
    	summary_ind.append(index)

    	if labels[index] != current_label:
    		current_label = labels[index]
    index += 1
cnt = Counter(summary.split(" "))
#print(cnt.most_common(30))
print()
#print(summary)

top30_dict = []
for i in range(cl):
	word_tokens = word_tokenize(str(getStrings(df_results[df_results["label"] == i]["results"])))
	filtered_sentence = [w for w in word_tokens if not w in stop_words]
	filtered_punct = [w for w in filtered_sentence if len(w) > 1]
	#split_it = filtered_sentence.split()
	cnt = Counter(filtered_punct)
	#bow.fit_transform(string_dict[i])
	#ordered =sorted(bow.vocabulary_.items(), key=lambda v:v[1], reverse=True)
	top30_dict.append(cnt.most_common(30))
	print(top30_dict[i])
	print()

top30_strings = []
for i in range(cl):
	 ' '.join(top30_dict[i])




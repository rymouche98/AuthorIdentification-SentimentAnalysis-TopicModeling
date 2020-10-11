import os
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold,GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier


REPLACE_NO_SPACE = re.compile("[.;:\',\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


def processing_text(text,use_stem, stop_words, remove_num=True,lower=False):
    text = REPLACE_NO_SPACE.sub("",text)
    text = REPLACE_WITH_SPACE.sub(" ", text)
    text = text.split(' ')
    if lower :
        text = [s.lower() for s in text]

    if remove_num :
        text = [word for word in text if not word.isdigit()]

    if stop_words:
        text = [s for s in text if s not in stopwords.words('english')]
    if use_stem :
        stemmer = PorterStemmer()
        text = [stemmer.stem(w) for w in text]
    return ' '.join(text)

def load_files(repo,use_stem,stopwords,remove_num,lower):
    files = os.listdir(repo)
    data = []
    for filename in files:
        path = os.path.join(repo, filename)
        file = open(path,'r')
        lines = file.readlines()
        lines = ' '.join(word for word in lines)
        data.append(processing_text(lines,use_stem,stopwords,remove_num,lower))
    return data

def load_data(path,use_stem, stopwords,remove_num,lower):
    pospath = os.path.join(path,'pos')
    negpath = os.path.join(path,'neg')
    datapos = load_files(pospath,use_stem,stopwords,remove_num,lower)
    dataneg = load_files(negpath,use_stem,stopwords,remove_num,lower)
    labelspos = [1] * len(datapos)
    labelsneg = [-1] * len(dataneg)
    data = datapos + dataneg
    labels = labelspos + labelsneg
    indices = np.random.permutation(np.arange(len(data)))
    tmp_data = []
    tmp_labels = []
    for indice in indices:
        tmp_data.append(data[indice])
        tmp_labels.append(labels[indice])
    data = tmp_data
    labels = tmp_labels
    return data, labels

def process_data(train_path,vectorizer,use_stem, stopwords,ngrams,lower=False,remove_num=True,max_feat = None):
    train_data, train_labels = load_data(train_path, use_stem, stopwords,remove_num,lower)
    if vectorizer == 'tfidf':
        vectorizer = TfidfVectorizer(min_df = 0.1, max_df = 0.9,sublinear_tf = True,use_idf = True,ngram_range=ngrams,max_features=max_feat)
        X_train = vectorizer.fit_transform(train_data)
    elif vectorizer == 'count':
        vectorizer = CountVectorizer(binary=False,ngram_range=ngrams,max_features=max_feat)
        X_train = vectorizer.fit_transform(train_data)
    else:
        vectorizer = CountVectorizer(binary=True,ngram_range=ngrams,max_features=max_feat)
        X_train = vectorizer.fit_transform(train_data)
    return X_train,train_labels, vectorizer

def write_pred(predictions,classifier):
    with open("result_sentiment_analysis"+classifier+".txt", "w") as file:
        for pred in predictions:
            file.write(pred + "\n")

def convert_pred(predictions):
    converted_pred=[]
    for p in predictions:
        if(p==-1):
            converted_pred.append('C')
        else:
            converted_pred.append('M')
    return converted_pred

def load_data_test(fileName,lemm,stopw,numbers):
    data=[]
    file = open(fileName,"r")
    for document in file:
        data.append(processing_text(document,use_stem=lemm,stop_words=stopw,remove_num=numbers))
    return data

def regularization_tuning(X_train,labels,max_iter=5000):
    C_values = [0.01, 0.02, 0.05,0.08, 0.1, 0.25, 0.35, 0.5, 1]
    scoresLR = []
    scoresSVM = []
    print("C \t\t Logistic Regression \t\t Linear SVM")
    for c in C_values:
        clfLR = LogisticRegression(C=c, max_iter=8000)
        clfSVM = LinearSVC(C=c,max_iter=8000)
        clfSVM.fit(X_train,labels)
        clfLR.fit(X_train,labels)
        scoreSVM = np.mean(cross_val_score(clfSVM, X_train, labels, cv=5))
        scoreLR = np.mean(cross_val_score(clfLR, X_train, labels, cv=5))
        scoresSVM.append(scoreSVM)
        scoresLR.append(scoreLR)
        print("{} \t\t\t {} \t\t\t {}".format(c, round(scoreLR,3), round(scoreSVM,3)))
    return C_values, scoresLR, scoresSVM
def print_words(clfSVM, clfLR, clfNB, vectorizer,verbose=False):
    feature_to_coefSVM = { word: coef for word, coef in zip(
        vectorizer.get_feature_names(), np.around(clfSVM.coef_[0],decimals=3)
        )
    }
    feature_to_coefLR = {
        word: coef for word, coef in zip(
            vectorizer.get_feature_names(), np.around(clfLR.coef_[0],decimals=3)
        )
    }
    feature_to_coefNB = {
        word: coef for word, coef in zip(
            vectorizer.get_feature_names(), np.around(clfNB.coef_[0],decimals=3)
        )
    }
    positiveSVM = np.asarray(sorted(feature_to_coefSVM.items(), key=lambda x: x[1], reverse=True)[:10])
    positiveLR = np.asarray(sorted(feature_to_coefLR.items(), key=lambda x: x[1], reverse=True)[:10])
    positiveNB = np.asarray(sorted(feature_to_coefNB.items(), key=lambda x: x[1], reverse=True)[:10])

    negativeSVM = np.asarray(sorted(feature_to_coefSVM.items(), key=lambda x: x[1], reverse=False)[:10])
    negativeLR = np.asarray(sorted(feature_to_coefLR.items(), key=lambda x: x[1], reverse=False)[:10])
    negativeNB = np.asarray(sorted(feature_to_coefNB.items(), key=lambda x: x[1], reverse=False)[:10])
    if verbose:
        print("------------------- 10 most positive words ----------------------\n")
        print("- Logistic Regression :\n {}\n".format(positiveLR[:,0]))
        print("- SVM:\n {}\n".format(positiveSVM[:,0]))
        print("- Naive Bayes:\n {}".format(positiveNB[:,0]))
        print("\n\n")
        print("-------------------- 10 most negative words ---------------------\n")
        print("- Logistic Regression :\n {}\n".format(negativeLR[:,0]))
        print("- SVM:\n {}\n".format(negativeSVM[:,0]))
        print("- Naive Bayes:\n {}".format(negativeNB[:,0]))
    return positiveSVM, negativeSVM, positiveLR, negativeLR

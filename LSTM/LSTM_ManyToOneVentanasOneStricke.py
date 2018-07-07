from __future__ import print_function

import os
import numpy as np

w2ind = dict()
 
file_name='emb39-word2vec'
fname_txt = file_name + '.txt'
fname_npy = file_name + '.npy'
npy = np.load(fname_npy)
txt = open(fname_txt,encoding="utf8").read().splitlines()  
for ind, wd in enumerate(txt):
    w2ind[wd] = ind

def w2v(wd):
    default_vector = np.zeros(len(npy[0]))
    try:
        return npy[w2ind[wd]]
    except KeyError:
        return default_vector

    wd2vect = w2v
    wd2vect.name = file_name
    wd2vect.dim = len(npy[0])
    return wd2vect

import numpy as np
import pandas as pd
import keras 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# -- Keras Import
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing import image

from keras.datasets import imdb
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN

from keras.layers import Activation, TimeDistributed, RepeatVector
from keras.callbacks import EarlyStopping, ModelCheckpoint

from pymongo import MongoClient

import numpy as np
import json;
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import numpy as np

from pymongo import MongoClient
client = MongoClient()
client = MongoClient("localhost",27017)
client = MongoClient("mongodb://localhost:27017/")
db = client["proyectodb"]

db.collection_names()


emojisMujeres = db['emojis_mujeres']
emojisHombres = db['emojis_hombres']


def obtenerFeatures():
    collHombresConcat= db.tweets_hombres_concat_url_cambiada.find();
    collMujeresConcat= db.tweets_mujeres_concat_url_cambiada.find();
    features_columns = [] #Lista que solo va a contener las palabras ordenados por la clave de menor a mayor
    cantidadUsuarios = 7000;
    tweets = []
    vClasificacion = []
    for i in range(0,cantidadUsuarios):
        try:   
            tweets = tweets + [collHombresConcat[i]['text']]        
            vClasificacion = vClasificacion + [0]
            tweets = tweets + [collMujeresConcat[i]['text']]        
            vClasificacion = vClasificacion + [1]
        except Exception as e:
            print(e)   

    #OBTENEMOS LOS FEATURES FILTRANDO POR OCURRENCIA
    from sklearn.feature_extraction.text import TfidfVectorizer
    ##vectorizer = TfidfVectorizer(min_df =0.04, max_df = 0.50)
    vectorizer = TfidfVectorizer(min_df =0.2, max_df = 0.80)
    vX_train = vectorizer.fit_transform(tweets)
    res = { vectorizer.vocabulary_[k]: k for k in vectorizer.vocabulary_} #Damos vuelta el diccionario que los indices sean la clave

    for i in range(0,len(res)):
        features_columns += [res[i]]

    return features_columns;



def obtenerListaPalabras(colHombres, colMujeres,usuarios, yUsuarios):
    listasPalabras = [] 
    listasPalabrasRes = [] 
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('spanish'))
    stop_words.update(['/','#', 'com' ,'www','http','https','.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 
    for i in range(0,len(usuarios)):
        cursor = colMujeres.find({"screen_name":{"$in":[usuarios[i]['screen_name']]}})
        cursorEm = emojisMujeres.find({"screen_name":{"$in":[usuarios[i]['screen_name']]}})
        cantEmojis = 0;
        for em in cursorEm:
            emojisMujer = em['emojis']
            cantEmojis = len(emojisMujer)			
       	elem = []
       	for c in cursor:			
            elem = c['text']
            res = tokenizer.tokenize(elem)
            res = [i for i in tokenizer.tokenize(elem) if i not in stop_words and i in features_columns]
            for r in res:    
                listasPalabras += [r]
            listasPalabras = listasPalabras[0:cantEmojis]
            if len(listasPalabras) > 0:
                listasPalabrasRes += [listasPalabras] 
            listasPalabras = [] 

        cursor = colHombres.find({"screen_name":{"$in":[usuarios[i]['screen_name']]}})
        cursorEm = emojisHombres.find({"screen_name":{"$in":[usuarios[i]['screen_name']]}})
        cantEmojis = 0;
        for em in cursorEm:
            emojisHombre = em['emojis']
            cantEmojis = len(emojisHombre)
       	elem  = []
        for c in cursor[0:cantEmojis]:          
            elem = c['text']
            res = tokenizer.tokenize(elem)
            res = [i for i in tokenizer.tokenize(elem) if i not in stop_words and i in features_columns]
            for r in res:        
                listasPalabras += [r]
				
            listasPalabras = listasPalabras[0:cantEmojis]
            listasPalabrasRes += [listasPalabras]	
            listasPalabras = []
    return listasPalabrasRes;

features_columns = obtenerFeatures()
concatMujeres = db['tweets_mujeres_concat_url_cambiada']
concatHombres = db['tweets_hombres_concat_url_cambiada']

json_usuariosTrain =open('C:\\proyecto\\informe\\usuariosTrain.json').read()
json_usuariosTest=open('C:\\proyecto\\informe\\usuariosTest.json').read()
json_yUsuariosTrain =open('C:\\proyecto\\informe\\yUsuariosTrain.json').read()
json_yUsuariosTest=open('C:\\proyecto\\informe\\yUsuariosTest.json').read()

usuariosTrain = json.loads(json_usuariosTrain)
usuariosTest = json.loads(json_usuariosTest)
yUsuariosTrain = json.loads(json_yUsuariosTrain) 
yUsuariosTest = json.loads(json_yUsuariosTest) 

yUs_train = yUsuariosTrain
yUs_test = yUsuariosTest
US_train = obtenerListaPalabras(concatHombres,concatMujeres,usuariosTrain, yUsuariosTrain)
US_test = obtenerListaPalabras(concatHombres,concatMujeres,usuariosTest, yUsuariosTest)

print("Us Train: " + str(len(US_train)))
print("y Us Train: " + str(len(yUs_train)))

print("Us Test: " + str(len(US_test)))
print("y Us test: " + str(len(yUs_test)))

data_dim = 300
timesteps = 9
ceros = np.zeros(data_dim)
lx_train = []
y_train = []
lugar = 0;

print("US_TRAIN")
print(US_train[0:1])
ventana = 10
for i in US_train:  
    lx_trainI = []
    cant = 0;         
    for j in  i: 
        x_train_vec=w2v(j)
        if( not(np.array_equal(x_train_vec,ceros))):
            lx_trainI += [x_train_vec]
        else:
            cant += 1
    if (len(lx_trainI)> 0):
        ini = 0
        fin = ventana - 1
        while(fin < len(lx_trainI)):
            lx_train += [lx_trainI[ini:fin]]
            ini = ini + 1
            fin = fin+ 1
            y_train += [yUs_train[lugar]]
    lugar +=1
    
x_train=np.array(lx_train)   

model = Sequential()

model.add(LSTM(1, batch_input_shape=(1, timesteps, data_dim), activation='hard_sigmoid' ))
model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=1, verbose=1, epochs=5)

ceros = np.zeros(data_dim)
lx_test = []
y_test = []
lugar = 0;
for i in US_test:
    lx_testI = []
    cant = 0;
    for j in  i: 
        x_test_vec=w2v(j)
        if( not(np.array_equal(x_test_vec,ceros))):
            lx_testI += [x_test_vec]
        else:
            cant += 1
    if (len(lx_testI)> 0):
        ini = 0
        fin = ventana - 1
        while(fin < len(lx_testI)):
            lx_test += [lx_testI[ini:fin]]
            ini = ini + 1
            fin = fin+ 1
            y_test += [yUs_train[lugar]]
    lugar +=1     

    
x_test=np.array(lx_test)   

score = model.evaluate(x_test, y_test, 1)

print(model.metrics_names)

print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#Prediccion
from sklearn import metrics   
def performance_clasificador (y_test, yp, avg): 
    #avg =weighted
    
    #Medida de Accuracy
    print ("Accuracy:\t\t",metrics.accuracy_score(y_test,yp))
    #Medida de Precision
    print ("Precision:\t\t",metrics.precision_score(y_test,yp,average=avg))
    #Medida de Recall
    print ("Recall:\t\t\t",metrics.recall_score(y_test,yp,average=avg))
    #Medida F1
    print ("Medida F:\t\t",metrics.f1_score(y_test,yp,average=avg))  
    #Matriz de Confusion
    print ("Matriz de Confusion:\t") 
    print (metrics.confusion_matrix(y_test,yp))
    return       


yp= model.predict(x_test,1)

#Make the output binary
for i in range(0, yp[:,0].size):
    for j in range(0, yp[0].size):
        if yp[i][j] >= 0.5:
            yp[i][j] = 1
        else:
            yp[i][j] = 0
performance_clasificador (y_test, yp, 'weighted')

## Importaciones
from __future__ import print_function
import json;
import numpy as np
import pandas as pd
import keras 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing import image

from keras.datasets import imdb
from keras.datasets import mnist

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, Concatenate
from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN

from keras.layers import Activation, TimeDistributed, RepeatVector
from keras.callbacks import EarlyStopping, ModelCheckpoint

from pymongo import MongoClient
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
from keras.models import Model    
from keras.layers import *

import os
import numpy as np
import json

####################################OBTENES TODOS LOS USUARIOS FILTROS POR EL CARGAR USUARIOS####################################################################
##SE CARGAN LOS USUARIOS ANTES DE CLCULAR LOS VECTORES DE PALABRAS YA SEPARADOS EN TRAIN Y TEST

json_usuariosTrain =open('C:\\proyecto\\NotebookParaClasificarConEmojis\\usuariosTrain.json').read()
json_usuariosTest=open('C:\\proyecto\\NotebookParaClasificarConEmojis\\usuariosTest.json').read()
json_yUsuariosTrain =open('C:\\proyecto\\NotebookParaClasificarConEmojis\\yUsuariosTrain.json').read()
json_yUsuariosTest=open('C:\\proyecto\\NotebookParaClasificarConEmojis\\yUsuariosTest.json').read()

usuariosTrain = json.loads(json_usuariosTrain)
usuariosTest = json.loads(json_usuariosTest)
yUsuariosTrain = json.loads(json_yUsuariosTrain) 
yUsuariosTest = json.loads(json_yUsuariosTest) 

##########################################################################################################################################################################

##################FUNCION WORD TO VECTOR####################################################################################################
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
    #print wd, ' not found in vecset'
        return default_vector

    wd2vect = w2v
    wd2vect.name = file_name
    wd2vect.dim = len(npy[0])
    return wd2vect
################################################################################################################################################

##################FUNCION EMOJI TO VECTOR####################################################################################################
#OBTENEMOS EMOJIS DE UN TWEET
import emot

def ObtenerListaEmojis(stringTweet): #Funci'on para obtener la lista de emojis
    emojis = []
    listaJsonEmoji = emot.emoji( stringTweet)
    for json in listaJsonEmoji:
        emojis += [json['value']]
    return emojis
	
import gensim.models as gsm

e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)

def ObtenerListaVectoresUsuario(listaEmojis):    
    listaVectores = []
    count = 0
    for em in listaEmojis:
        try:
            vector = e2v[em]    # Produces an embedding vector of length 300
            listaVectores += [vector]
        except: #en caso de no existir se devuelve un vector de 0s de tamanio 300
            count += 1
    return listaVectores       
	
import gensim.models as gsm

e2v = gsm.KeyedVectors.load_word2vec_format('emoji2vec.bin', binary=True)

def ObtenerListaVectoresUsuario(listaEmojis):    
    listaVectores = []
    count = 0
    for em in listaEmojis:
        try:
            vector = e2v[em]    # Produces an embedding vector of length 300
            listaVectores += [vector]
        except: #en caso de no existir se devuelve un vector de 0s de tamanio 300
            count += 1
    return listaVectores       



def ObtenerListaDeListaDeEmojisUsuarios(colHombres, colMujeres,usuarios, yUsuarios):
    #Obtenemos una lista que contenga todos los emojis de cada usuario y ademas armamos el y
    listaEmojis = [] 
    y = []
    
    for i in range(0,len(usuarios)):
       	cursor = colMujeres.find({"screen_name":{"$in":[usuarios[i]['screen_name']]}})
       	emojisMujer= []
       	for c in cursor:
            emojisMujer = c['emojis']
            vectores = ObtenerListaVectoresUsuario(emojisMujer)
            listaEmojis += [vectores]            
      
       	cursor = colHombres.find({"screen_name":{"$in":[usuarios[i]['screen_name']]}})
       	emojisHombre = []
        for c in cursor:
            emojisHombre = c['emojis']
            vectores = ObtenerListaVectoresUsuario(emojisHombre)
            listaEmojis += [vectores]
    print("---------------------------------")	
    print("Cant listaEmojis: " +str(len(listaEmojis)))
    print("---------------------------------")	
    return listaEmojis;
	
#######################################################################################################################################################


#############################FUNCIONES PARA OBTENER LA LISTA DE VECTORES DE PALABRAS ####################################################################


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
            if len(listasPalabras) > 0:
                listasPalabrasRes += [listasPalabras]	
            listasPalabras = []
    return listasPalabrasRes;

#######################################################################################################################################################
	
	

#############################OBTENEMOS LOS VECTORES DE LOSEMOJIS DE LOS TWEETS Y SEPARAMOS EN TRAIN Y TEST ############################################################
#Conexión a BD
client = MongoClient()
# alternatively..
client = MongoClient("localhost",27017)
client = MongoClient("mongodb://localhost:27017/")
db = client["proyectodb"]

db.collection_names()

emojisMujeres = db['emojis_mujeres']
emojisHombres = db['emojis_hombres']

yUS_train_emojis = yUsuariosTrain
yUS_test_emojis= yUsuariosTest
US_train_emojis  = ObtenerListaDeListaDeEmojisUsuarios(emojisHombres,emojisMujeres, usuariosTrain, yUsuariosTrain)
US_test_emojis = ObtenerListaDeListaDeEmojisUsuarios(emojisHombres,emojisMujeres, usuariosTest, yUsuariosTest)

#########################################################################################################################################################


#############################OBTENEMOS LOS VECTORES DE PALABRAS DE LOS TWEETS Y SEPARAMOS EN TRAIN Y TEST ###################################################################
features_columns = obtenerFeatures()
concatMujeres = db['tweets_mujeres_concat_url_cambiada']
concatHombres = db['tweets_hombres_concat_url_cambiada']
yUs_train = yUsuariosTrain
yUs_test = yUsuariosTest
US_train = obtenerListaPalabras(concatHombres,concatMujeres,usuariosTrain, yUsuariosTrain)
US_test = obtenerListaPalabras(concatHombres,concatMujeres,usuariosTest, yUsuariosTest)
##################################################################################################################################################


###########################################LSTM DE LOS EMOJIS###########################################################################################
data_dim = 300
timesteps = 12
batch_size= 13

model_emojis = Sequential()
model_emojis.add(LSTM(1, batch_input_shape=(1, timesteps, data_dim), activation='hard_sigmoid', dropout = 0.2))

ceros = np.zeros(data_dim)
lx_train_emoji = []
y_train_emoji = []
lugar = 0;
for listaEmoji in US_train_emojis: #Para cada lista de vectores de emojis de un usuario    
    lx_train_emojiI = []
    cant = 0;        
    for j in  listaEmoji: #Para cada vector de la lista de vecores
        x_train_emoji_vec=j
        if( not(np.array_equal(x_train_emoji_vec,ceros))):
            lx_train_emojiI += [x_train_emoji_vec]
        else:
            cant += 1
    if (len(lx_train_emojiI)> 0):
        ini = 0
        fin = 0
        while((fin + batch_size) <= len(lx_train_emojiI)):
            f = fin+batch_size-1
            lx_train_emoji += [lx_train_emojiI[ini:f]]
            ini = fin+batch_size
            fin = fin+batch_size
            y_train_emoji += [yUS_train_emojis[lugar]]
            
        if (fin != len(lx_train_emojiI)):
            cant_ceros= batch_size - (len(lx_train_emojiI) - fin)
            f = len(lx_train_emojiI) -1
            res_lx_train_emoji = lx_train_emojiI[fin:f] 
            res_lx_train_emoji += [ceros] * cant_ceros
            lx_train_emoji+= [res_lx_train_emoji]
            y_train_emoji += [yUS_train_emojis[lugar]]
    lugar +=1


print("***************TRAIN EMOJIS******************")   
x_train_emoji=np.array(lx_train_emoji)      
print(x_train_emoji.shape)    

## TEST

ceros = np.zeros(data_dim)
lx_test_emoji = []
y_test_emoji = []
lugar = 0;

for listaEmoji in US_test_emojis: #Para cada lista de vectores de emojis de un usuario    
    lx_test_emojiI = []
    cant = 0;        
    for j in  listaEmoji: #Para cada vector de la lista de vecores
        x_test_emoji_vec=j
        if( not(np.array_equal(x_test_emoji_vec,ceros))):
            lx_test_emojiI += [x_test_emoji_vec]
        else:
            cant += 1
    if (len(lx_test_emojiI)> 0):
        ini = 0
        fin = 0
        while((fin + batch_size) <= len(lx_test_emojiI)):
            f = fin+batch_size-1
            lx_test_emoji += [lx_test_emojiI[ini:f]]
            ini = fin+batch_size
            fin = fin+batch_size
            y_test_emoji += [yUS_test_emojis[lugar]]
            
        if (fin != len(lx_test_emojiI)):
            cant_ceros= batch_size - (len(lx_test_emojiI) - fin)
            f = len(lx_test_emojiI) -1
            res_lx_test_emoji = lx_test_emojiI[fin:f] 
            res_lx_test_emoji += [ceros] * cant_ceros
            lx_test_emoji+= [res_lx_test_emoji]
            y_test_emoji += [yUS_test_emojis[lugar]]
    lugar +=1
    
    
x_test_emoji=np.array(lx_test_emoji)   

print("***************TEST EMOJIS******************")  
print(x_test_emoji.shape)    

ceros = np.zeros(data_dim)
lx_train = []
y_train = []
lugar = 0;

print("######################################################")
ventana = 13
for i in US_train:
    lx_trainI = []         
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
    
x_train=np.array(lx_train[0:x_train_emoji.shape[0]])   

print("lx_train SHAPE PALABRAS")    
print(x_train.shape) 
                 
#Se crea LSTM palabras
model_palabras = Sequential()
model_palabras.add(LSTM(1, batch_input_shape=(1, timesteps, data_dim), activation='hard_sigmoid', dropout = 0.2))

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
    
x_test=np.array(lx_test[0:x_test_emoji.shape[0]])   

print("tamaño TEST palabras")    
print(x_test.shape)    

#Prediccion
from sklearn import metrics   
def performance_clasificador (vX_test, y_test, clf, avg, label): 
    #avg =weighted
    yp = clf.predict(vX_test)
    #Medida de Accuracy
    print ("Accuracy:\t\t",metrics.accuracy_score(y_test,yp))
    #Medida de Precision
    print ("Precision:\t\t",metrics.precision_score(y_test,yp,average=avg, pos_label=label))
    #Medida de Recall
    print ("Recall:\t\t\t",metrics.recall_score(y_test,yp,average=avg, pos_label=label))
    #Medida F1
    print ("Medida F:\t\t",metrics.f1_score(y_test,yp,average=avg, pos_label=label))  
    #Matriz de Confusion
    print ("Matriz de Confusion:\t") 
    print (metrics.confusion_matrix(y_test,yp))
    return  

##################################################################################################################################################

################################################# MERGE #################################################################################################
from keras import metrics

merge = Concatenate()([model_emojis.output, model_palabras.output])

out = Dense(1, activation='hard_sigmoid')(merge)

modelall = Model([model_emojis.input, model_palabras.input], out)
modelall.compile(optimizer='adam', loss='binary_crossentropy', metrics= [metrics.categorical_accuracy])
modelall.fit([x_train_emoji, x_train], y_train_emoji,batch_size=1, verbose=1, epochs=1)

score = modelall.evaluate([x_test_emoji, x_test], y_test_emoji,1)
 

print("%s: %.2f%%" % (modelall.metrics_names[1], score[1]*100))
#########################################################################################################################################################
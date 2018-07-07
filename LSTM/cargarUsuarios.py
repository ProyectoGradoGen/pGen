import json;
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import numpy as np
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

client = MongoClient()
client = MongoClient("localhost",27017)
client = MongoClient("mongodb://localhost:27017/")
db = client["proyectodb"]

db.collection_names()

concatMujeres = db.tweets_mujeres_concat_url_cambiada.find()
concatHombres = db.tweets_hombres_concat_url_cambiada.find()
def obtenerListaUsuarios():
	cantidadUsuarios = 7000;
	modUsurios = 150
	listaUsuarios=[]
	yUsuarios=[]
	for i in range(0,cantidadUsuarios):
		try:   
			if(i%modUsurios == 0):
				jsonMujer={'screen_name': concatMujeres[i]['screen_name'], 'text':concatMujeres[i]['text']}
				listaUsuarios += [jsonMujer]
				yUsuarios += [1]
				jsonHombre={'screen_name': concatHombres[i]['screen_name'], 'text':concatHombres[i]['text']}
				listaUsuarios += [jsonHombre]
				yUsuarios += [0]
		except Exception as e:
			print(e)   
	return listaUsuarios, yUsuarios
listaUsuario, yUsuarios = obtenerListaUsuarios()
json.dump(listaUsuario, open('C:\proyecto\informe\listaUsuario.json', 'w') )
json.dump(yUsuarios, open('C:\proyecto\informe\yUsuarios.json', 'w') )

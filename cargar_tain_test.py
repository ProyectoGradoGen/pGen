
import json;
from sklearn.model_selection import train_test_split

##OBTENES TODOS LOS USUARIOS FILTROS POR EL CARGAR USUARIOS
json_listaUsuarios =open('C:\\proyecto\\NotebookParaClasificarConEmojis\\listaUsuario.json').read()
json_y=open('C:\\proyecto\\NotebookParaClasificarConEmojis\\yUsuarios.json').read()

json_listaUsuarios = json.loads(json_listaUsuarios)
json_y = json.loads(json_y)

##DIVIMOS LOS USUARIOS ANTES DE CLCULAR LOS VECTORES DE PALABRAS
usuariosTrain,usuariosTest,yUsuariosTrain,yUsuariosTest = train_test_split(json_listaUsuarios,json_y,test_size=0.25, random_state=5)

json.dump(usuariosTrain, open('C:\\proyecto\\NotebookParaClasificarConEmojis\\usuariosTrain.json', 'w') )
json.dump(usuariosTest, open('C:\\proyecto\\NotebookParaClasificarConEmojis\\usuariosTest.json', 'w') )
json.dump(yUsuariosTrain, open('C:\\proyecto\\NotebookParaClasificarConEmojis\\yUsuariosTrain.json', 'w') )
json.dump(yUsuariosTest, open('C:\\proyecto\\NotebookParaClasificarConEmojis\\yUsuariosTest.json', 'w') )

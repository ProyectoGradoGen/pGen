
import json;
from sklearn.model_selection import train_test_split

##OBTENES TODOS LOS USUARIOS FILTROS POR EL CARGAR USUARIOS
json_listaUsuarios =open('C:\proyecto\informe\listaUsuario.json').read()
json_y=open('C:\proyecto\informe\yUsuarios.json').read()

json_listaUsuarios = json.loads(json_listaUsuarios)
json_y = json.loads(json_y)

print("json_listaUsuarios: " + str(len(json_listaUsuarios)))
print("json_y: " + str(len(json_y)))
##DIVIMOS LOS USUARIOS ANTES DE CLCULAR LOS VECTORES DE PALABRAS
usuariosTrain,usuariosTest,yUsuariosTrain,yUsuariosTest = train_test_split(json_listaUsuarios,json_y,test_size=0.25, random_state=5)


json.dump(usuariosTrain, open('C:\\proyecto\\informe\\usuariosTrain.json', 'w') )
json.dump(usuariosTest, open('C:\\proyecto\\informe\\usuariosTest.json', 'w') )
json.dump(yUsuariosTrain, open('C:\\proyecto\\informe\\yUsuariosTrain.json', 'w') )
json.dump(yUsuariosTest, open('C:\\proyecto\\informe\\yUsuariosTest.json', 'w') )
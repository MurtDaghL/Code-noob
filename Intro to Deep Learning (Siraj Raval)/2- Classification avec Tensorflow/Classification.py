# D'après la vidéo de Siraj Raval: https://www.youtube.com/watch?v=4urPuRoT1sE&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3&index=4

import pandas as pd, numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from time import sleep

# Importation et suppression des données inutiles:
data = pd.read_csv("D:\\Domanis\\GitHub\\Code-noob\\Intro to Deep Learning (Udacity)\\2- Classification avec Tensorflow\\data.csv")
data.drop(['index','price','sq_price'],axis=1,inplace=True)
data = data[0:10]

# Ajout des labels (totalement arbitraire)
data.loc[:, ('y1')] = [1,1,1,0,1,0,1,1,1,1]
data.loc[:, ('y2')] = (data['y1'] == 0)
data.loc[:, ('y2')] = data['y2'].astype(int)
print(data)

# Conversion en tenseurs:
InputX = data.loc[:,['area','bathrooms']].as_matrix()
InputY = data.loc[:,['y1','y2']].as_matrix() 

# Hyperparamètres:
lr = 0.000001
cycles = 1000
affichage = 50
nb_exemples = InputY.size


# Création du réseau neuronal:
    
    # Création de l'entrée
x = tf.placeholder(tf.float32, [None,2]) # Les placeholders sont les "portes" du réseau <=> Les dimensions du tenseur d'entrée et de sortie
    # Création des coefficients synaptiques:
W = tf.Variable(tf.zeros([2,2])) # Les variables contiennent et mettent à jour les paramètres
    # Ajout des biais (exemple: b est le biais de l'équation ax+b):
b = tf.Variable(tf.zeros([2]))

    # Propagation:
y_valeur = tf.add(tf.matmul(x,W),b) # matmul <=> multiplication de matrices
    # Application de la fonction d'activation softmax:
y = tf.nn.softmax(y_valeur)
    # Création de la sortie
y_ = tf.placeholder(tf.float32, [None,2])

    # Apprentissage:

    # Création de la fonction d'erreur (mse):
erreur = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*nb_exemples)   # reduce_sum <=> Somme des éléments du tenseur
    # Création de l'algorithme d'apprentissage (rétropropagation du gradient):
optimiseur = tf.train.GradientDescentOptimizer(lr).minimize(erreur)



# Initialisation des variables et de la session tensorflow:
init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)


# Boucle d'apprentissage:
for i in range(cycles):
    session.run(optimiseur,feed_dict={x: InputX, y_: InputY})
    if (i % affichage) == 0:
        cc = session.run(erreur,feed_dict={x: InputX, y_: InputY})
        print('Etape:','%04d' % (i), 'Erreur = ', '{:.9f}'.format(cc))

print("Optimisation terminée ! ")
print('Erreur finale = ', '{:.9f}'.format(cc), 'W=', session.run(W), 'b=', session.run(b))
sleep(10)




# D'après la vidéo de Siraj Raval: https://www.youtube.com/watch?v=XdM6ER7zTLk&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3&index=2

from numpy import *
from pandas import *
from time import sleep
import matplotlib.pyplot as plt

def run():
    # Importation des données:
    points = read_csv("D:\\Domanis\\GitHub\\Code-noob\\Intro to Deep Learning (Udacity)\\1- Régression linéaire\\data.csv", names=['heures de travail','notes /100'])
    print(points.head())
    points = array(points)

    # Définition des hyperparamètres:
    taux = 0.0001
    cycles = 100
    # y = ax+b
    init_a = 1.32243102276
    init_b = 7.99102098225
    [a,b] = repet_gradient_descent(points, init_a, init_b, taux, cycles)
    
    def f(x): return a*x+b

    plt.scatter(points[:,0],points[:,1])
    plt.plot(points[:,0],[f(x) for x in points[:,0]])
    plt.show()


def repet_gradient_descent(points, a_depart, b_depart, taux, nb_cycles):
    a = a_depart
    b = b_depart
    
    for i in range(nb_cycles):
        a,b = etape_gradient(a, b, array(points), taux)
    return [a,b]

# Rétropropagation du gradient:
def etape_gradient(a, b, points, taux):
    gradient_a = 0
    gradient_b = 0
    N = len(points)
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        gradient_b += -(2/N) * (y-(a*x+b))   #Utilisation de la dérivée de l'erreur en fonction de a
        gradient_a += -(2/N) * x * (y-(a*x+b))   #Utilisation de la dérivée de l'erreur en fonction de b
    a -= (taux*gradient_a)
    b -= (taux*gradient_b)
    print(str(a) + 'x + ' + str(b) )
    return [a,b]
   

def calcul_erreur(a, b, points):
    erreur_totale = 0
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        erreur_totale += (y-(a*x+b))**2    # Somme des erreurs au carré (pour avoir uniquement des valeurs positives)
    return erreur_totale / len(points)

if __name__ == '__main__':
    run()
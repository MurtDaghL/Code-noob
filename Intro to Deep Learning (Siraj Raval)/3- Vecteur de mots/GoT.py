# D'après la vidéo de Siraj Raval: https://www.youtube.com/watch?v=pY9EwZ02sXU&index=6&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3

from codecs import open as c_open # Utile pour lire le fichier texte en utf-8
from glob import glob # Utile pour lister les éléments du dossier data
import multiprocessing # Utile pour accélérer l'apprentissage
import pprint
from nltk.tokenize import sent_tokenize, word_tokenize # Utile pour "tokenizer" le corpus
from nltk.corpus import stopwords # Utile pour filtrer les mots inutiles
import gensim.models.word2vec as w2v # Bibiliothèque avec le modèle w2v + de nombreuses fonctions utiles en rapport
import numpy as np # Calcul mathématique (dépendance de bcp de bibliothèques)
import pandas as pd # Permet de stocker les données dans un DataFrame
import pickle # Utile pour "sérialiser" les données:
from os.path import exists # Utile pour vérifier l'existence des fichier .pickle
import sklearn.manifold # Utile pour réduire la dimensionalité des données (compression dans un plan 2D)
import matplotlib.pyplot as plt # Traçe des graphiques, dépendance de seaborn
import seaborn as sns # Permet de traçer des graphiques et de les exploiter facilement
from time import sleep # Utile pour mettre en pause le script
# nltk.download() # A dé-commenter pour télécharger tout les modèles de nltk

if not exists('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus.pickle'):
    print("\n\nI/ IMPORTATION DES LIVRES ET CREATION DU CORPUS:\n\n")
    sleep(1)

    liste_fichiers = sorted(glob('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/data/*'))
    corpus = ''
    for livre in liste_fichiers:
        print('Lecture de {0}'.format(livre))
        with c_open(livre, 'r','utf-8') as fichier:
            corpus += fichier.read()
        print('Le corpus est désormais long de {0} caractères\n'.format(len(corpus)))

    pickle.dump(corpus,open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus.pickle","wb"))
    print("Enregistrement du corpus terminé")

    


if not exists('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus_token.pickle'):
    print("\n\nII/ TOKENIZATION DU CORPUS :\n\n")
    sleep(1)

    print("Importation du corpus...")
    corpus = pickle.load(open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus.pickle","rb"))
    print('Transformation du corpus en liste de phrases...')
    corpus = sent_tokenize(corpus)
    new_corpus = []

    print("Transformations des phrases en liste de mots et suppression des mots indésirables...")
    sleep(1)
    i = 0
    r = 0
    mots_indésirables = set(stopwords.words("english"))
    for phrase in corpus:
        i += 1 
        liste_de_mots = word_tokenize(phrase)
        liste_de_mots_filtrée = []
        for mot in liste_de_mots:
            if mot not in mots_indésirables:
                liste_de_mots_filtrée.append(mot)
            else: r +=1
        new_corpus.append(liste_de_mots_filtrée)
        print("{0} phrases ajoutées, {1} mots rejetés".format(i,r))

    pickle.dump(new_corpus,open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus_token.pickle","wb"))
    print("Enregistrement du corpus tokenizé terminé")
    del new_corpus




if not exists("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/model.w2v"):
    print('\n\nIII/ CONSTRUCTION ET ENTRAINEMENT DU MODELE DE WORD2VEC :\n\n')
    sleep(1)

    print("Importation du corpus tokenizé...")
    corpus = pickle.load(open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus_token.pickle","rb"))
    print("Le corpus contient {} phrase".format(len(corpus)))

    nbworkers = multiprocessing.cpu_count()
    seed = 1 # Seed pour générateur de nombres pseudo-aléatoire
    N = 300 #Dimension des vecteurs de mots
    nb_mots_min = 3
    taille_contexte = 7
    downsampling = 1e-3 # Paramètre de "downsample" pour les mots fréquents

    model = w2v.Word2Vec(seed=seed, workers=nbworkers, size=N, min_count=nb_mots_min, window=taille_contexte, sample=downsampling)
    print("Construction du vocabulaire...")
    model.build_vocab(corpus)
    print("Entraînement du modèle word2vec...")
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/model.w2v")

    print("Entraînement et enregistrement terminé !")


if not exists("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/TSNE 2D.pickle"):
    print('\n\nIV/ CONSTRUCTION ET ENTRAINEMENT DU MODELE TSNE (Projection 2D):\n\n')
    sleep(1)

    print("Chargement du modèle w2v...")
    model = w2v.Word2Vec.load('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/model.w2v')
    print("Récupération des vecteurs de mots du modèle...")
    matrice_vecteur_mots = model.wv.syn0 #Récupère les vecteurs de mots
    print("Apprentissage du tsne...")
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0, verbose=1) # Réduction de la dimensionalité (TSNE pour t-stochastic distributed neighbor embedding)
    matrice_vecteur_mots_2D = tsne.fit_transform(matrice_vecteur_mots,) #Entraînement du tsne (algo. Machine Learning)

    pickle.dump(matrice_vecteur_mots_2D,open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/TSNE 2D.pickle","wb"))
    print("Apprentissage du tsne et enregistrement des vecteurs de mots réduits en 2D terminé !")



print("\n\nV/ VISUALISATION ET EXPLOITATION DU MODELE:\n\n")
sleep(1)
print("Chargement du modèle w2v et des vecteur de mots réduits...")
model = w2v.Word2Vec.load('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/model.w2v')
matrice_vecteur_mots_2D = pickle.load(open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/TSNE 2D.pickle","rb"))

print("Création du DataFrame regroupant tout les mots...")
points = pd.DataFrame( [ (mot,coord[0],coord[1]) for mot, coord in [ (mot, matrice_vecteur_mots_2D[model.wv.vocab[mot].index]) for mot in model.wv.vocab ] ], columns=["mot","x","y"] )
print(points.head(10))


sns.set_context("poster")
# graph = points.plot.scatter("x", "y", s=10, figsize=(20,12))

def zoom(x,y):
    slice = points[
        (x[0] <= points.x) &
        (points.x <= x[1]) & 
        (y[0] <= points.y) &
        (points.y <= y[1]) ]
    graph = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        graph.text(point.x + 0.005, point.y + 0.005, point.mot, fontsize=11)
    plt.show()

zoom((0,30),(0,30))


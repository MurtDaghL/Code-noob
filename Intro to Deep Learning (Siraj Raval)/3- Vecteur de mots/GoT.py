
from codecs import open as c_open # Utile pour lire le fichier texte en utf-8
from glob import glob # Utile pour lister les éléments du dossier data
import multiprocessing # Utile pour accélérer l'apprentissage
import pprint
import re
from nltk.tokenize import sent_tokenize, word_tokenize # Utile pour "tokenizer" le corpus:
import gensim.models.word2vec as w2v # Bibiliothèque avec le modèle w2v + de nombreuses fonctions utiles en rapport
import numpy as np # Calcul mathématique (dépendance de bcp de bibliothèques)
import pandas as pd # Permet de stocker les données dans un DataFrame
import pickle # Utile pour "sérialiser" les données:
from os.path import exists # Utile pour vérifier l'existence des fichier .pickle
import sklearn.manifold # Utile pour réduire la dimensionalité des données (compression dans un plan 2D)
import matplotlib.pyplot as plt # Traçe des graphiques, dépendance de seaborn
import seaborn as sns # Permet de traçer des graphiques et de les exploiter facilement
 
# nltk.download()



# IMPORTATION DES LIVRES ET CREATION DU CORPUS:
if not exists('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus.pickle'):
    liste_fichiers = sorted(glob('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/data/*'))
    print(liste_fichiers)
    corpus = ''
    for livre in liste_fichiers:
        print('Lecture de {0}'.format(livre))
        with c_open(livre, 'r','utf-8') as fichier:
            corpus += fichier.read()
        print('Le corpus est désormais long de {0} caractères'.format(len(corpus)))
    print("IMPORTATION TERMINEE !")

    pickle.dump(corpus,open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus.pickle","wb"))
    print("Enregistrement du corpus terminé")


# TOKENIZATION :
if not exists('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus_token.pickle'):
    corpus = pickle.load(open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus.pickle","rb"))
    print("Chargement du corpus terminé !")
    # Séparation du corpus en phrases:
    corpus = sent_tokenize(corpus)
    new_corpus = []
    i = 0
    # Séparation des phrases en mots:
    for phrase in corpus:
        i += 1
        new_corpus.append(word_tokenize(phrase))
        print("{} phrases ajoutées".format(i))
    pickle.dump(new_corpus,open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus_token.pickle","wb"))
    print("Enregistrement du corpus tokenizé terminé")


corpus = pickle.load(open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/corpus_token.pickle","rb"))
print("Chargement du corpus tokenizé terminé !")
print("Le corpus contient {} phrase".format(len(corpus)))


# CONSTRUCTION ET ENTRAINEMENT DU MODELE DE WORD2VEC :
if not exists("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/model.w2v"):
    nbworkers = multiprocessing.cpu_count()
    seed = 1 # Seed pour générateur de nombres pseudo-aléatoire
    N = 300 #Dimension des vecteurs de mots
    nb_mots_min = 3
    taille_contexte = 7
    downsampling = 1e-3 # Paramètre de "downsample" pour les mots fréquents

    model = w2v.Word2Vec(seed=seed, workers=nbworkers, size=N, min_count=nb_mots_min, window=taille_contexte, sample=downsampling)

    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/model.w2v")
    print("Entraînement et enregistrement terminé !")


# VISUALISATION:
model = w2v.Word2Vec.load('D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/3- Vecteur de mots/model.w2v')
print("CHARGEMENT DU MODELE TERMINE !")
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0) # Réduction de la dimensionalité (TSNE pour t-stochastic distributed neighbor embedding)
matrice_vecteur_mots = model.wv.syn0 #Récupère les vecteurs de mots
print("Apprentissage du tsne...")
matrice_vecteur_mots_2D = tsne.fit_transform(matrice_vecteur_mots) #Entraînement du tsne (algo. Machine Learning)

points = pd.DataFrame( [ (mot,coord[0],coord[1]) for mot, coord in [ (mot, matrice_vecteur_mots_2D[model.vocab[mot].index]) for mot in model.vocab ] ], columns=["mot","x","y"] )
points.head(10)

sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20,12))

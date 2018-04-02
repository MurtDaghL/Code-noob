
# D'après la vidéo de Siraj Raval: https://www.youtube.com/watch?v=18adykNGhHU&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3&index=8
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import time
from sklearn.externals import joblib
from os.path import exists
import pickle

if not exists('D:\\Domanis\\GitHub\\Code-noob\\Intro to Deep Learning (Siraj Raval)\\4 - Système de recommandation\\song_df.pickle'):
    print("\nI/ TELECHARGEMENT DES DONNEES:\n")
    triplets_file = 'https://static.turi.com/datasets/millionsong/10000.txt' # Triplets: ID utilisateur /  ID Musique / Nombre d'écoutes
    songs_metadata_file = 'https://static.turi.com/datasets/millionsong/song_data.csv' #Métadata des musiques
    # Triplets
    song_df_1 = pd.read_table(triplets_file,header=None)
    song_df_1.columns = ['user_id', 'song_id', 'listen_count']
    # Métadata
    song_df_2 =  pd.read_csv(songs_metadata_file)
    # Fusion des deux datasframes:
    song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")
    # Enregistrement:
    pickle.dump(song_df,open('D:\\Domanis\\GitHub\\Code-noob\\Intro to Deep Learning (Siraj Raval)\\4 - Système de recommandation\\song_df.pickle','wb'))
    print("Enregistrement terminé !")
    
print("Importation des données...")
song_df = pickle.load(open("D:/Domanis/GitHub/Code-noob/Intro to Deep Learning (Siraj Raval)/4 - Système de recommandation/song_df.pickle","rb"))
print("Taille des données: {} lignes".format(len(song_df)))
time.sleep(10)

# Sélection d'un sous-ensemble de données:
song_df = song_df.head(10000)
# Fusion du nom de l'artiste et du nom de la chanson:
song_df["song"] = song_df.title.map(str) + " - " + song_df["artist_name"]

# Regroupement et tri des musiques suivant la popularité des chansons et ajout du pourcentage:
print("Traitement des données...\n")
song_df_triée = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
écoutes_totales = song_df_triée['listen_count'].sum()
song_df_triée['pourcentage %'] = song_df_triée['listen_count'].div(écoutes_totales)*100
song_df_triée.sort_values(['listen_count','song'], ascending = [0,1], inplace=True)
print("Musiques les plus populaire:\n")
print(song_df_triée.head(5))

utilisateurs = len(song_df.user_id.unique())
musiques = len(song_df.song.unique())
print("Nombres d'utilisateurs: {0} \nNombres de musiques: {1}".format(utilisateurs,musiques))



traindata, testdata = train_test_split(song_df, test_size = 0.20 , random_state = 0)


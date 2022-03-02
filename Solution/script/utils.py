import os
import pandas as pd
import pickle
import numpy as np

class DataClicks:
    '''
    Initialisation de la classe
    
    Parameters:
        interaction_path_dir (str): chemin vers le dossier contenant les interractions des utilisateurs
        metadata_path (str): chemin vers le fichier contenant les metadonnées des articles
        article_embedding_path (str) : chemin vers le fichier contenant la matrice embedding des articles
        
    Returns:
        None
    '''
    def __init__(self,interaction_path_dir : str = None,metadata_path :str = None, article_embedding_path : str = None):
        self.interaction_path_dir = interaction_path_dir
        self.metadata_path = metadata_path
        self.article_embedding_path = article_embedding_path
        
    def load_data(self):
        '''
        Chargement des différent fichiers sources en DataFrame et Numpy
        
        Parameters: 
            None
        Returns :
            interaction_df (pd.DataFrame): DataFrame de toutes les interaction utilisateur sur l'application
            metadata_df (pd.DataFrame): DataFrame des méta données des articles
            article_matrice_df (numpy.array): Plongement de données réalisé sur la base de 364047 articles.
        '''
        if self.interaction_path_dir :
            interaction_df = self._get_interaction()
        if self.metadata_path :
            metadata_df = self._get_metadata()
        if self.article_embedding_path :
            article_matrice = self._get_article_embedding()
            
        return interaction_df, metadata_df, article_matrice
    
    def _get_interaction(self):
        '''
        Chargement de tous les fichiers sources des interactions contenue dans le dossier clicks
        
        Parameters:
            None
            
        Returns:
            interaction_df (pd.DataFrame): interaction des utilisateurs
        '''
        interaction_df = pd.DataFrame()
        for f in os.listdir(self.interaction_path_dir):
            df = pd.read_csv(self.interaction_path_dir + f)
            interaction_df = interaction_df.append(df)
            
        interaction_df["session_start"] = pd.to_datetime(interaction_df["session_start"], unit="ms")
        interaction_df["click_timestamp"] = pd.to_datetime(interaction_df["click_timestamp"], unit="ms")
        return interaction_df
    
    def _get_metadata(self):
        '''
        Chargement du fichier des meta données des articles.
        
        Parameters:
            None
            
        Returns :
            metadata_df (pd.DataFrame): meta données des différents articles
        '''
        metadata_df=pd.read_csv(self.metadata_path)
        
        metadata_df["created_at_ts"] = pd.to_datetime(metadata_df["created_at_ts"],unit="ms")
        
        return metadata_df
        
    def _get_article_embedding(self):
        '''
        Chargement du fichier de plongement de données.
        
        Parameters:
            None
            
        Returns :
            article_matrice (numpy.array): matrice embedding des différents articles.
        '''
        open_click_pickle = open(self.article_embedding_path,"rb")
        article_matrice = pickle.load(open_click_pickle)
        open_click_pickle.close()
        return article_matrice

    
class ContentBased:
    
    def __init__(self):
        return


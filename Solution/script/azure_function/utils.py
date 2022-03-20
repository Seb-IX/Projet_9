import os
import pandas as pd
import pickle
import numpy as np

from scipy.spatial import distance
import scipy.sparse as sparse

import implicit

from random import randint

class DataClicks:
    '''
    Initialisation de la classe
    
    Parameters:
        interaction_path_dir (str): chemin vers le dossier contenant les interractions des utilisateurs
        metadata_path (str): chemin vers le fichier contenant les metadonnées des articles
        article_embedding_path (str) : chemin vers le fichier contenant la matrice embedding des articles
        final_df_path (str) : chemin vers le fichier final de modélisation (CF)
        
    Returns:
        None
    '''
    def __init__(self,interaction_path_dir : str = None,metadata_path :str = None, article_embedding_path : str = None, final_df_path : str = None):
        self.interaction_path_dir = interaction_path_dir
        self.metadata_path = metadata_path
        self.article_embedding_path = article_embedding_path
        self.final_df_path = final_df_path
        
        self.interact_load = False
        self.metadata_load = False
        self.embedding_load = False
        self.final_df_load = False
        
    def load_data(self):
        '''
        Chargement des différent fichiers sources en DataFrame et Numpy
        
        Parameters: 
            None
        Returns :
            interaction_df (pd.DataFrame): DataFrame de toutes les interaction utilisateur sur l'application
            metadata_df (pd.DataFrame): DataFrame des méta données des articles
            article_matrice_df (numpy.array): Plongement de données réalisé sur la base de 364047 articles.
            final_df ((pd.DataFrame): fichier final de modélisation créer
        '''
        # interact data
        if self.interaction_path_dir:
            interact_df = self.get_interaction_data()
        else:
            final_df = None
        # metadata 
        if self.metadata_path:
            metadata_df = self.get_metadata()
        else:
            final_df = None
        # embedding matrix    
        if self.article_embedding_path:
            article_matrice = self.get_embedding()
        else:
            final_df = None
        # final dataframe 
        if self.final_df_path:
            final_df = self.get_final_df()
        else:
            final_df = None
            
        return interact_df, metadata_df, article_matrice, final_df
    
    def _load_interaction_data(self):
        '''
        Chargement de tous les fichiers sources des interactions contenue dans le dossier clicks
        
        Parameters:
            None
            
        Returns:
            None
        '''
        self.interaction_df = pd.DataFrame()
        for f in os.listdir(self.interaction_path_dir):
            df = pd.read_csv(self.interaction_path_dir + f)
            self.interaction_df = self.interaction_df.append(df)
            
        self.interaction_df["session_start"] = pd.to_datetime(self.interaction_df["session_start"], unit="ms")
        self.interaction_df["click_timestamp"] = pd.to_datetime(self.interaction_df["click_timestamp"], unit="ms")
                
    def get_interaction_data(self):
        '''
        Getter du fichier metadata
        
        Parameters:
            None
            
        Returns :
            interaction_df (pd.DataFrame): interaction des utilisateurs
        '''
        if not self.interact_load:
            self._load_interaction_data()
        return self.interaction_df
    
    def _load_metadata(self):
        '''
        Chargement du fichier des meta données des articles.
        
        Parameters:
            None
            
        Returns :
            None
        '''
        self.metadata_df=pd.read_csv(self.metadata_path)
        
        self.metadata_df["created_at_ts"] = pd.to_datetime(self.metadata_df["created_at_ts"],unit="ms")

    
    def get_metadata(self):
        '''
        Getter du fichier metadata
        
        Parameters:
            None
            
        Returns :
            metadata_df (pd.DataFrame): meta données des différents articles
        '''
        if not self.metadata_load:
            self._load_metadata()
        return self.metadata_df
        
    def _load_embedding(self):
        '''
        Chargement du fichier de plongement de données.
        
        Parameters:
            None
            
        Returns :
            None
        '''
        open_click_pickle = open(self.article_embedding_path,"rb")
        self.embedding = pickle.load(open_click_pickle)
        open_click_pickle.close()
    
    def get_embedding(self):
        '''
        Getter du fichier embedding
        
        Parameters:
            None
            
        Returns :
            article_matrice (numpy.array): matrice embedding des différents articles.
        '''
        if not self.embedding_load:
            self._load_embedding()
        return self.embedding
    
    def _load_final_df(self):
        '''
        Chargement du fichier final de modélisation.
        
        Parameters:
            None
            
        Returns :
            None
        '''
        self.final_df = pd.read_csv(self.final_df_path)
    
    def set_final_df(self,path : str, load_after_set = False):
        '''
        Setter du fichier final utilisé pour les modélisation
        
        Parameters:
            path (str) : Chemin vers le fichier df
            
        Returns :
            article_matrice (numpy.array): matrice embedding des différents articles.
        '''
        self.final_df_path = path
        self.final_df_load = False
        self.final_df = None
        if load_after_set:
            self._load_final_df()
        
    def get_final_df(self):
        '''
        Getter du fichier final utilisé pour les modélisation
        
        Parameters:
            None
            
        Returns :
            article_matrice (numpy.array): matrice embedding des différents articles.
        '''
        if not self.final_df_load:
            self._load_final_df()
        return self.article_matrice

    
############## Model Recommendation ################
class ModelRecommendation(DataClicks):
    def __init__(self, interaction_path_dir : str = None, metadata_path :str = None, article_embedding_path : str = None, final_df_path : str = None):
        DataClicks.__init__(self,interaction_path_dir = interaction_path_dir,
                            metadata_path = metadata_path,
                            article_embedding_path = article_embedding_path,
                            final_df_path = final_df_path)
    
    def train_(self):
        raise Exception("Initialize train method...")
        
    def recommend_(self,userid : int, N : int = 5):
        '''
        Recommandation
        
        Parameters:
            user_id (int): identifiant du user pour la recommandation
            N (int optional): nombre d'article à recommander
            
        Returns :
            arr (list-> int): liste des identifiants d'articles recommandé
        '''
        raise Exception("Initialize recommend method...")
    
    def recommend_list_user_(self,list_user : list,N : int =5):
        all_recommend = {}
        for ids in list_user:
            all_recommend[ids] = self.recommend_(ids,N=N)
        return all_recommend
    
################# content-based ################
class ContentBasedRecommandation(ModelRecommendation):
    '''
    Initialisation de la classe
    
    Parameters:
        path_user_interaction_directory (str): chemin vers le dossier contenant les interractions des utilisateurs
        path_emdedding (str) : chemin vers le fichier contenant la matrice embedding des articles
        
    Returns:
        None
    '''
    def __init__(self,path_emdedding : str, path_user_interaction_directory : str):
        
        ModelRecommendation.__init__(self,article_embedding_path = path_emdedding, 
                                     interaction_path_dir = path_user_interaction_directory)
        ModelRecommendation._load_embedding(self)
        ModelRecommendation._load_interaction_data(self)
        self.interaction_df = self.interaction_df[["user_id","click_article_id"]]
            
    def recommend_(self,user_id : int,N : int = 5):
        '''
        Recommandation
        
        Parameters:
            user_id (int): identifiant du user pour la recommandation
            N (int optional): nombre d'article à recommander
            
        Returns :
            arr (list-> int): liste des identifiants d'articles recommandé
        '''
        embedding_temp = self.embedding
        # on récupére tout les article lu par l'utilisateurs
        var = self.interaction_df.loc[self.interaction_df['user_id']==user_id]['click_article_id'].tolist()
        # on choisi un article simillaire au article lu par l'utilisateurs aléatoirement
        if len(var) == 0:
            # Si il n'a pas encore lu d'article on lui propose par rapport à l'article le plus populaire
            value = self.interaction_df.groupby("click_article_id")["click_article_id"].size().sort_values(ascending=False).index[0]
        else:
            value = randint(0, len(var))
        
        # On supprime les article déjà lu par l'utilisateur
        for i in range(0, len(var)):
            if i != value:
                embedding_temp = np.delete(embedding_temp, [i], 0)

        arr = []

        # on supprime l'article selectionné
        f = np.delete(embedding_temp, [value], 0)

        # on récupére les n articles les plus similaire à celui selectionné
        for i in range(0, N):
            # On récupére la matrice de distance
            distances = distance.cdist([embedding_temp[value]], f, "cosine")[0]
            min_index = np.argmin(distances)
            f = np.delete(f, [min_index], 0)
            result = np.where(self.embedding==f[min_index])
            arr.append(result[0][0])

        return arr
    
################# collaboratif-filtering ################ 
class ContentCollaboratifFiltering(ModelRecommendation):
    '''
    Initialisation de la classe
    
    Parameters:
        final_df_path (str) : Chemin vers le fichier final de modélisation
        
    Returns:
        None
    '''
    def __init__(self,final_df_path : str):
        ModelRecommendation.__init__(self,final_df_path=final_df_path)
        
        ModelRecommendation._load_final_df(self)
      
        self.interact_article_df = self.final_df.groupby(["user_id","article_id"]).size().to_frame().reset_index().rename(columns={0:"interaction_article"})
        self.sparse_user_item = sparse.csr_matrix((self.interact_article_df['interaction_article'].astype(float),
                                                 (self.interact_article_df['user_id'], self.interact_article_df['article_id'])))

            
    def train_(self,factors=100,iterations=200,regularization=0.1,alpha=1.0,show_progress=True):
        '''Entrainement de la modélisation ALS (AlternatingLeastSquares)
        
        Parameters : 
            factors (int) : Le nombre de facteurs latents à calculer, default = 100
            iterations (int): Le nombre d'itérations ALS à utiliser lors de l'ajustement des données, default = 200.
            regularization (float) : Le facteur de régularisation à utiliser, default = 0.1
            alpha (float) : Augmente ou diminue le poid des valeurs de la matrice user_item csr
            show_progress (bool) : Affichage de la progression 
        Retunrs:
            None
        '''
        self.model = implicit.als.AlternatingLeastSquares(factors=factors,iterations=iterations,regularization=regularization)
        self.model.fit(self.sparse_user_item*alpha, show_progress=show_progress)
    
    def recommend_(self,userid : int,N : int =5, get_simi_score = False):
        try :
            if get_simi_score:
                return self.model.recommend(userid, self.sparse_user_item[userid],N=N)
            else:
                return self.model.recommend(userid, self.sparse_user_item[userid],N=N)[0].tolist()
        except IndexError:
            raise Exception('User ID not validat.')
    
    
################# Evaluate-model-CF ################
class EvaluateRS(DataClicks):
    '''
    Initialisation de la classe
    
    Parameters:
        path_user_interaction_directory (str): chemin vers le dossier contenant les interractions des utilisateurs
        metadata_path (str) : chemin vers le fichier des metadonnées article
        
    Returns:
        None
    '''
    def __init__(self,path_user_interaction_directory,metadata_path,
                 ceil_all = pd.to_datetime("2017-10-17 00:00:00"),
                 ceil_split = pd.to_datetime("2017-10-10 00:00:00")):
        # Chargement des fichier
        DataClicks.__init__(self,interaction_path_dir = path_user_interaction_directory,metadata_path = metadata_path)
        DataClicks._load_interaction_data(self)
        DataClicks._load_metadata(self)
        
        self.metadata_df["anciennete"] = abs(self.metadata_df['created_at_ts'] - ceil_split)
        self.metadata_df['anciennete'] = self.metadata_df['anciennete'] / np.timedelta64(1, 'D')
        df_old = self.interaction_df.merge(self.metadata_df,left_on="click_article_id",right_on="article_id")
        df_old.drop(df_old[df_old["created_at_ts"] > df_old["click_timestamp"]].index,axis=0,inplace=True)
        
        # On reinitialise l'anciennete avec la dernière valeur que l'on gardera
        self.metadata_df["anciennete"] = abs(self.metadata_df['created_at_ts'] - ceil_all)
        self.metadata_df['anciennete'] = self.metadata_df['anciennete'] / np.timedelta64(1, 'D')
        df_new = self.interaction_df.merge(self.metadata_df,left_on="click_article_id",right_on="article_id")
        df_new.drop(df_new[df_new["created_at_ts"] > df_new["click_timestamp"]].index,axis=0,inplace=True)
        
        
        self.old_interaction = df_old[df_old["click_timestamp"] <= ceil_split].copy()
        self.new_interaction = df_new[df_new["click_timestamp"] <= ceil_all].copy()
        
        self.final_test_df = self._transform_data(self.new_interaction)
        self.final_train_df = self._transform_data(self.old_interaction)
        

    def _transform_data(self,dataframe):
      
        rating = dataframe[['click_article_id', 'user_id']].groupby(["click_article_id"],as_index=False).agg("count")
        rating.rename(columns={"user_id":"interaction"},inplace=True)
        rating.rename(columns={"click_article_id":"click_article_id_inter"},inplace=True)

        final_df_train = dataframe.merge(rating,how="left",left_on=["article_id"],right_on=["click_article_id_inter"])
        # on retire les columns inutile
        final_df_train.drop(["session_id","click_article_id","session_start","session_size","click_environment",
                             "click_deviceGroup","click_os","click_country","click_region",
                             "click_referrer_type","click_article_id_inter","publisher_id"],axis=1,inplace=True)
        # on remplace les NA par des 0 pour les futurs traitement
        final_df_train["interaction"] = final_df_train["interaction"].fillna(0)
        return final_df_train
    
    def evaluate_user(self, recommendations : list,userid : int):
        '''Méthode pour l\'évaluation d\'un seul utilisateur
        
        Parameters:
            recommendations (list -> int):
            userid

        Returns:
            final_score (float): 
            count_same_cat (int): 
            anciennete_mean (float): 
            count_same_article_read (int):
        '''
        new_list = self.new_interaction[self.new_interaction["user_id"] == userid]["article_id"].tolist()
        old_list = self.old_interaction[self.old_interaction["user_id"] == userid]["article_id"].tolist()

        for i in old_list:
            new_list.remove(i)

        count_same_article_read = 0
        count_article_new = 0
        for ids in new_list:
            count_article_new += 1
            if ids in recommendations:
                count_same_article_read+=1

        all_cat = self.new_interaction[self.new_interaction["article_id"].isin(new_list)].groupby("article_id")["category_id"].last().values.tolist()
        all_cat_recommandation = self.new_interaction[self.new_interaction["article_id"].isin(recommendations)].groupby("article_id")["category_id"].last().values.tolist()
        count_same_cat = sum([all_cat.count(i) for i in all_cat_recommandation])

        anciennete_mean = self.metadata_df[self.metadata_df["article_id"].isin(recommendations)].groupby("article_id")["anciennete"].last().values.mean()

        return count_same_cat, anciennete_mean, count_same_article_read, count_article_new
    
    def evaluate_(self,recommandations : map,list_user : list):
        '''Méthode pour l\'évaluation d\'un échantillion d\'utilisateur
        
        Parameters:
            recommandation (map -> {id:list} -> int) : list des recommandation d'article 
        Returns:
            count_same_cat (int): 
            anciennete_mean (float): 
            count_same_article_read (int):
        '''
        self.count_cat = 0
        self.anciennete_mean = []
        self.count_same_article = 0
        self.count_article_new_read = 0
        for userid in list_user:
            recommandation = recommandations[userid]
            count, anciennete, count_same_article_read, count_article_new = self.evaluate_user(recommandation, userid)
            self.count_article_new_read += count_article_new  
            self.count_cat += count 
            self.anciennete_mean.append(anciennete)
            self.count_same_article+=count_same_article_read
            
        return self.count_cat, self.anciennete_mean, self.count_same_article
    
    def display_result(self,count_same_article_display=True,round_day=3,round_cat=3):
        print(f"L\'ancienneté des articles proposés sont d'environ {round((sum(self.anciennete_mean) / len(self.anciennete_mean)),round_day)} jours.")
        print(f"Il y a un total de {self.count_cat} catégories simillaire soit {round((self.count_cat / self.count_article_new_read) * 100,round_cat)}%.")
        if count_same_article_display:
            print(f"Et le nombre d'article recommandé qui à était vraiment lu est de {self.count_same_article} ")
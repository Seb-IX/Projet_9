
import logging

import azure.functions as func
from utils import ContentBasedRecommandation
from azure.storage.blob import BlobServiceClient,  __version__
import os
import joblib
import pickle


########### CHARGEMENT DES FICHIERS ###############
file_download = "based_model.joblib"

try:
    ## Récupération des fichiers dans le blob storage Azure (compte de stockage)
    logging.info("Azure Blob Storage v" + __version__ + " - Python quickstart sample")

    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    download_file_path = "./data/"+ file_download

    blob_client = blob_service_client.get_blob_client(container="data", blob=file_download)

    with open(download_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    
    model_based = joblib.load("./data/based_model.joblib")

except Exception as ex:
    logging.info('Exception:')
    logging.info(ex)


########## HTTP TRIGGER ###########
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    userId = req.params.get('userId')

    if not userId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            userId = req_body.get('userId')
    
    result = model_based.recommend_(int(userId))
    
    return func.HttpResponse(f"{result}")


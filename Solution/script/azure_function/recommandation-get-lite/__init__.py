
import logging

import azure.functions as func

import pickle

map_recommandation_file = open("./data/recommandation_cf.pickle","rb")
recommandation = pickle.load(map_recommandation_file)
map_recommandation_file.close()

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
    
    result = recommandation[int(userId)]  
    
    return func.HttpResponse(f"{result}")


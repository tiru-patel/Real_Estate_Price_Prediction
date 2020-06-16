import json
import pickle
import numpy as np
__locations=None
__data_columns = None
__model=None

def estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index=-1
    x1 = np.zeros(len(__data_columns))
    x1[0] = sqft
    x1[1] = bath
    x1[2] = bhk
    if loc_index >= 0:
        x1[loc_index] = 1
    return round(__model.predict([x1])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts.....start")
    global __data_columns
    global __locations
    global __model
    with open("./artifacts/columns.json",'r') as f:
        __data_columns=json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./artifacts/bangalore_home_price_model.pickle",'rb') as f:
        __model=pickle.load(f)
    print("Loading artifacts.... done")


if __name__=='__main__':
    load_saved_artifacts()

    print(get_location_names())
    print(estimated_price('1st Phase JP Nagar',1000,3,3))

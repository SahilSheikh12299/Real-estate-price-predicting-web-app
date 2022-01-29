import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None



def get_estimated_price(location,sqft,bhk,bath):

    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    

    """
    We are initialising a numpy array of zeros, having the same length as that of the columns.json file, 
    now, the values that we get as an input i.e., the location, sqft, bhk, bath, we are storing those values in our array
    at the respective indices. After that, we are setting the value of our location to be 1 in our numpy array by finding
    its index in our columns tableand that setting that value to be 1 at that index in our numpy array.
    Rest of the values are zero. In this way, we have all the things with us to predict the price, now we just have to supply 
    this array and get the output!

    """

    x = np.zeros(len(__data_columns))           
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    #x[3] to the last number are all locations


    if loc_index >= 0:
        x[loc_index] = 1

    #We get a 2-d array as an output of the predict method, thats why we are using [0] index to only give us the price

    return round(__model.predict([x])[0],2)


def get_location_names():
    return __locations


def load_saved_artifacts():

    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    

    with open("./server/artifacts/columns.json","r") as f:
        __data_columns = json.load(f)['data_columns']      
        __locations = __data_columns[3:]

    global __model
    if __model is None:
        with open("./server/artifacts/banglore_home_prices_model.pickle","rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...Done!")



if __name__ == '__main__':

    load_saved_artifacts()
    print(get_location_names())
    #print(get_estimated_price('1st Phase JP naGar',2500,4,5))
    #print(get_estimated_price('Nehru Nagar',2000,4,5))
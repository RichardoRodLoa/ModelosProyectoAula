import pandas as pd
from sklearn.model_selection import train_test_split


from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"
FILTERED_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "filtereddataset.csv"

def get_data():
    '''
    retorna los datos originales del dataframe alojado en el archivo ./data/dataset.csv
    en forma de X e y

    returns
        X: la matriz de datos (10000, 49)
        y: array de variable objetivo (, 49)
    '''

    df = pd.read_csv(DATA_PATH)

    arr = df.to_numpy()
    arr = arr[::,1::] # eliminar la primera columna correspondiente al id

    # Preparacion de los datos para el entrenamiento
    X = arr[::,0:48]
    y = arr[::,48]
    return (X, y)


def get_data_partitioned(test_size=0.2): 
    ''' 
    retorna los datos particionados apartir del archivo ./data/dataset.csv
    haciendo uso de la funcion `get_data()`
    
    returns 
    - X_train: matriz x de entrenamiento 
    - X_test: matriz x de validacion
    - y_train: variable objetivo de entrenamiento
    - y_test: variable objetivo de validacion
    '''

    X, y = get_data()
    
    return train_test_split(X, y, test_size=test_size)


def get_filtered_data(): 
    '''
    retorna los datos filtrados como resultado del analisis del notebook `data_analysis.ipynb`
    
    returns 
    - X nd.array (10000, 41)
    - y nd.array (1000,)
    '''
    
    df = pd.read_csv(FILTERED_DATA_PATH, index_col="id")
    
    X = df.drop(columns="CLASS_LABEL").to_numpy()
    y = df['CLASS_LABEL'].to_numpy()
    return (X, y)

    
def get_filtered_data_partitioned(test_size=0.2): 
    '''
    retorna los datos particionados del dataframe de datos filtrados
    
    returns: 
    - X_train
    - X_test, 
    - y_train
    - y_test
    '''
    
    X, y = get_filtered_data()
    
    return train_test_split(X, y, test_size=test_size)
    
    
if __name__ == "__main__": 
    X, y = get_filtered_data()
    print(f'shape X: {X.shape}')
    print(f'shape y: {y.shape}')

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# gather data

boston_dataset = load_boston()
data = pd.DataFrame(data = boston_dataset.data , columns = boston_dataset.feature_names)
data.head()

features = data.drop(["INDUS","AGE"], axis = 1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns = ["PRICE"])

CRIME_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8

ZILLOW_MEAN_PRICE = 715.469
SCALE_FACTOR = 715.468 / np.median(boston_dataset.target)

property_stats = features.mean().reshaper(1,11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river = False,
                    high_confidence = True):
    
    #Configure Property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river :
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
            
        
    
    #Make Prediction
    log_estimate = regr.predict(property_stats)[0][0]
    
    #Calc Range 
    if high_confidence :
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95

    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE 
        interval = 68

    return log_estimate, upper_bound , lower_bound , interval 

def get_dollar_estimate(rm,ptratio, chas = False , large_range = True) :
    """Estimate price of a house in Boston
    rm - rooms in propery
    ptratio - pupils per teacher
    chas - is it next to chas river, True or False
    """
    if rm<1 or ptratio > 30:
        print("That is unrealistic, please try again")
        return
    
    log_est, upper , lower , conf = get_log_estimate(rm, ptratio, 
                                                      next_to_river = chas,
                                                      high_confidence = large_range)
#convert to current doller
    dollar_est =np.e**log_est * 1000 * SCALE_FACTOR
    dollar_high =np.e**upper * 1000 * SCALE_FACTOR
    dollar_low =np.e**lower * 1000 * SCALE_FACTOR
#round dollar value to nearest 1000

    rounded_est = np.around(dollar_est, -3 )
    upper_est = np.around(dollar_high, -3)
    lower_est = np.around(dollar_low, -3)

    print(f"The estimated property value is {rounded_est}")
    print(f"At confidence {conf} the value range is :")
    print(f"${lower_est} at the lower end, ${upper_est} at the high end")
#FETCHING THE DATA FOR PREDICTIONS FROM THE DATABASE

from pymongo import MongoClient
import certifi
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime

#  Connect to the database and store the retrieved data into a variable
ca = certifi.where()
client = MongoClient("mongodb+srv://...",
                     tlsCAFile=ca)
                     
db = client.BigData_Project
mycol = db["weatherForPred"]
dontWant = {"_id":0}
dfFromMongo = pd.DataFrame(list(mycol.find({},dontWant)))

print(dfFromMongo)


# USING API TO FETCH THE NEXT 16 DAYS OF WEATHER DATA

import pandas as pd
import requests
import json
from datetime import datetime

# Order: NSW, NT, QL, SA, TA, VI, WA
coordinates = ["-31.2532;146.9211", "-19.4914;132.5510", "-22.5752;144.0848", "-30.0002;136.2092", "-37.7323;144.9578",
               "-36.9848;143.3906", "27.6728;121.6283"]
forecastDF = pd.DataFrame(columns=['Date', 'Region', 'Precipitation', 'RelativeHumidity', 'Temperature'])

i = 0
for coord in coordinates:
    latLong = coord.split(';')
    api_url = "https://api.weatherbit.io/v2.0/forecast/daily?lat=" + latLong[0] + "&lon=" + latLong[
        1] + "&key=be2f4cf779be47a8a31cf71b75aff8c7"
    req = requests.get(api_url)
    data = json.loads(req.text)
    for weather in data["data"]:
        if (i == 0):
            region = "NSW"
        elif (i == 1):
            region = "NT"
        elif (i == 2):
            region = "QL"
        elif (i == 3):
            region = "SA"
        elif (i == 4):
            region = "TA"
        elif (i == 5):
            region = "VI"
        elif (i == 6):
            region = "WA"
        else:
            raise Exception("An error has occurred, please fix!")

        wrongFormat = datetime.strptime(weather["datetime"], '%Y-%m-%d').date()
        date = datetime.strftime(wrongFormat, '%m/%d/%Y')
        humid = weather["rh"]
        temp = weather["temp"]
        precipitation = weather["precip"]
        dfData = [date, region, precipitation, humid, temp]
        tempDF = pd.DataFrame([dfData], columns=['Date', 'Region', 'Precipitation', 'RelativeHumidity', 'Temperature'])
        forecastDF = pd.concat([forecastDF, tempDF])
    i += 1


# PREPROCESSING THE DATA AND TRAINING THE MODEL

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

weatherDF = pd.read_csv("weatherForPred.csv", sep=",")
#weatherDF = dfFromMongo

weatherDF = weatherDF.replace(0,0.00000001)
weatherDF.drop_duplicates(inplace=True)
weatherDF = weatherDF.dropna(how='any')

X = weatherDF.drop(['Estimated_fire_area', 'SolarRadiation', 'WindSpeed', 'SoilWaterContent','Date','Region'], axis = 1).copy()

y = weatherDF['Estimated_fire_area'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

regr = linear_model.LinearRegression()


forecastDF.replace(0,0.00000001)
forecastDF.dropna(how='any')
forecastDF = forecastDF.drop(['Date','Region'], axis = 1).copy()

regr.fit(X_train, y_train)

forecastPredictions = regr.predict(forecastDF)
print(forecastPredictions)


# CREATING A DATAFRAME FROM THE PREDICTION DATA AND CREATING A CSV

import datetime
j = 0
region = 0
predDF = pd.DataFrame(columns=["Region","Date", "Prediction"])
regionName = ""
# Order: NSW, NT, QL, SA, TA, VI, WA

for i in forecastPredictions:
    if j < 16:
        if (region == 0):
            regionName = "NSW"
        elif (region == 1):
            regionName = "NT"
        elif (region == 2):
            regionName = "QL"
        elif (region == 3):
            regionName = "SA"
        elif (region == 4):
            regionName = "TA"
        elif (region == 5):
            regionName = "VI"
        elif (region == 6):
            regionName = "WA"
        else:
            raise Exception("An error has occurred")
        date = datetime.date.today() + datetime.timedelta(days=j)
        tempList = [regionName,date, i]
        tempDF = pd.DataFrame([tempList], columns=["Region", "Date", "Prediction"])
        predDF = pd.concat([predDF, tempDF])
    j += 1
    if j == 16:
        j = 0
        region += 1


predDF.to_csv(r'results.csv', index = False, header=True)


# CREATING A VISUALISATION OUT OF THE PREDICTION DATAFRAME AND MEASURING THE FITTING

import plotly.express as px

df2 = predDF.groupby('Region') \
       .agg({'Prediction':'mean'}) \
       .reset_index()

from sklearn.metrics import mean_squared_error, r2_score

y_test = y_test.head(112)
RMSE = np.sqrt(mean_squared_error(y_test, forecastPredictions))
rtwo = r2_score(y_test, forecastPredictions)

title = "Daily average estimated fire area in km2<br><br>RMSE: {} <br>r2: {}".format(RMSE,rtwo)

for template in ["plotly_dark"]:
    fig = px.treemap(df2,template=template,width=950, height=750, path=[ px.Constant("All"), 'Prediction', 'Region'], values='Prediction',
                  color='Prediction', color_continuous_scale='reds',title=title )
fig.update_traces(hovertemplate='Daily average estimated fire area in km2 = %{value}<extra></extra>')
fig.update_layout(
    title_font_size=12
)
fig.show()
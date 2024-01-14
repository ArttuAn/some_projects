""" 
******************************************************************************************************************************************************************************************************************************************************************************************
                                                                    *DATA DISCOVERY*
******************************************************************************************************************************************************************************************************************************************************************************************
"""

""" 
        *************************************************************************************************************************************************************************************
        *Reading the Kaggle dataset and storing it into a dataframe. First reading the negative reviews into another dataframe, then positive reviews into another one then concatting them *
        *************************************************************************************************************************************************************************************
"""

import pandas as pd

print("Reading the Kaggle dataset and storing it into a dataframe. First reading the negative reviews into another dataframe, then positive reviews into another one then concatting them ---->")

# Reading the dataset and storing it into a dataframe
hotel_reviews_complete = pd.read_csv('Hotel_Reviews.csv')
print("All the hotel reviews before choosing the right columns, as read straight from the csv: \n",hotel_reviews_complete)
# Reading the dataset and storing it into a dataframe filtering out the columns we dont need
hotel_reviews_neg = pd.read_csv('Hotel_Reviews.csv', usecols=['Negative_Review'])
hotel_reviews_neg = hotel_reviews_neg.rename(columns={'Negative_Review': 'Review'})
hotel_reviews_neg['Pos'] = 0

hotel_reviews_pos = (pd.read_csv('Hotel_Reviews.csv', usecols=['Positive_Review']))
hotel_reviews_pos= hotel_reviews_pos.rename(columns={'Positive_Review': 'Review'})
hotel_reviews_pos['Pos'] = 1

# Concatting the two dataframes, positive and negative ones, axis representing the axis along which the data is concatinated
hotel_reviews = pd.concat([hotel_reviews_pos, hotel_reviews_neg],axis=0)


print("<---- Reading the Kaggle dataset done, dataset stored in a dataframe")

""" 
                            *************************************************************************************
                            *Using Selenium to scrape a dataset from agoda and saving the data into a dataframe *
                            *************************************************************************************
"""

print("Using Selenium to scrape a dataset from Agoda and saving the data into a dataframe ---->")

# Importing the required libraries
from selenium import webdriver
driver = webdriver.Chrome(executable_path=r'C:\...')
# These libraries should remove few error messages
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = webdriver.Chrome(options=options)

reviewUrls = []
scrapedReviews=[]

with open('reviewUrls2.txt') as webSet:
    for url in webSet:
        reviewUrls.append(url.replace('\n', ''))

for url in reviewUrls:

    driver.get(url)
    container = driver.find_elements_by_class_name("Review-comment")

    for j in range(len(container)):

        pos = 0
        review = container[j].find_element_by_class_name("Review-comment-bodyText").text
        rating = int(float(container[j].find_element_by_class_name("Review-comment-leftScore").text))

        if rating > 5:  
            pos = 1
        if rating == 5:
            continue
        
        scrapedReviews.append([review,pos]) 

scrapedReviewsSelDF = pd.DataFrame(scrapedReviews, columns=['Review', 'Pos'])	

print(scrapedReviewsSelDF)

driver.quit()

print("<---- Using Selenium to scrape a dataset from agoda and saving the data into a dataframe done")

""" 
                            *************************************************************************************
                            *Using bs4 to scrape a dataset from tripadvisor and saving the data into a dataframe*
                            *************************************************************************************
"""

from bs4 import BeautifulSoup
import requests
import re

print("Using bs4 to scrape a dataset from Tripadvisor and saving the data into a DataFrame---->")

# Initializing headers variable so that the request can be made by imitating an individual
headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'}

dfPos = pd.DataFrame()
dfNeg = pd.DataFrame()
scrapedReviews = pd.DataFrame()

dfPosList = []
dfNegList = []
scrapedReviewsList = []
reviewUrls = []

with open('reviewUrls.txt') as webSet:
    for url in webSet:
        reviewUrls.append(url.replace('\n', ''))

for url in reviewUrls:
    print("Scraping a new page")
    getPage = requests.get(url,headers=headers) 
    statusCode = getPage.status_code 
    
    if(statusCode == 200):
        print("Succesfully scraped the page, code 200")

        soup = BeautifulSoup(getPage.text, 'html.parser')

        for item in soup.findAll('div', class_= "cWwQK MC R2 Gi z Z BB dXjiy"):

            comment = item.find('div', class_="pIRBV _T").span.text
            reviewScore = item.find('div', class_="emWez F1").span['class']
            posComment = 0
       
            if reviewScore[1] == "bubble_10" or reviewScore[1] == "bubble_20":
                dfNegList.append([comment,  posComment]) 
                dfNeg = pd.DataFrame(dfNegList, columns=['Review','Pos'])
                scrapedReviewsList.append([comment, posComment]) 
                scrapedReviews = pd.DataFrame(scrapedReviewsList, columns=['Review','Pos'])
          
            if reviewScore[1] == "bubble_40" or reviewScore[1] == "bubble_50":
                posComment = 1
                dfPosList.append([comment,  posComment]) 
                dfPos = pd.DataFrame(dfPosList, columns=['Review','Pos'])    
                scrapedReviewsList.append([comment,  posComment ]) 
                scrapedReviews = pd.DataFrame(scrapedReviewsList, columns=['Review','Pos'])    
              
all_reviews = pd.concat([scrapedReviews, hotel_reviews],axis=0)
all_reviews = pd.concat([scrapedReviewsSelDF, all_reviews],axis=0)
print(all_reviews)

print("<---- Succesfully scraped the hotel reviews and stored them in a dataframe and concatted them with the Kaggle dataset")

""" 
                                *************************************************************
                                *Writing own reviews and storing them into a dataframe after*
                                *************************************************************
"""
print("Writing own reviews and storing them into a dataframe after ---->")


Review1 = "Staying in this hotel made me sick for two weeks, still have not recovered"
Review2 = "Positively surprised of the service I received. I especially liked the front desk clerks"
Review3 = "As expected. Nothing fancy but not too bad either. Would recommend for bigger families. Not gonna stay again though"

first = { "Review": Review1, "Pos": 0 }
second = { "Review": Review2,  "Pos": 1 }
third = { "Review": Review3,  "Pos": 0 }

allReviews = [first,second,third]
ownReviewsDF = pd.DataFrame(allReviews)
all_reviews = pd.concat([all_reviews, ownReviewsDF],axis=0)
all_reviews = all_reviews.sample(frac=1).reset_index(drop=True)

#comment out this one later
all_reviews = all_reviews.head(1000)

print(all_reviews)
print("<---- Succesfully wrote own reviews and stored them in a dataframe and concatted them with the Kaggle dataset")
""" 
***********************************************************************************************************************************************************************************************************************************************************************************************
                                                           *DATA PREPARATION*
****************************************************************************************************************************************************************************************************************************************************************************
"""

""" 
                ******************************************************************************************************************************************
                *Storing the combined datasets that are in one dataframe into a database and then retrieving the data using parametrized stored procedure*
                ******************************************************************************************************************************************
"""
print("Storing the combined datasets that are in one dataframe into a database and then retrieving the data using parametrized stored procedure ---->")

from sqlalchemy import create_engine

engine = create_engine('mysql+mysqlconnector://root:root@localhost/reviews')

all_reviews.to_sql(name='hotel_reviews',con=engine,if_exists='replace',index=False, chunksize=1000) 

conn = engine.raw_connection()
cur=conn.cursor()
minWords = 15
cur.callproc('selectAllLongerThan', [minWords])
for row in cur.stored_results(): 
    results = row.fetchall()
    colNamesList=row.description
colNames=[i[0] for i in colNamesList]
result_dicts = [dict(zip(colNames, row)) for row in results]

df=pd.DataFrame(result_dicts)

conn.close()


print("Printing the dataframe from the DB:")
print(df)

print("<---- Succesfully retrieved the dataframe stored into the database and stored it in a dataframe")

""" 
                                            *******************
                                            *Cleaning the data* 
                                            *******************
"""
print("Cleaning the data/preprocessing ---->")

df["Review"] = df["Review"].apply(lambda text: text.lower().strip())
df["Review"] = df["Review"].apply(lambda text: re.sub('\W+',' ', text))

print("<---- Succesfully cleaned the data/preprocessed the dataframe")
""" 
                                                ********************
                                                *Creating bar charts* 
                                                ********************
"""
print("Creating bar charts ---->")

import matplotlib.pyplot as plt

ax = df['Pos'].value_counts().plot.bar(rot=0)
plt.show()

print("The distribution of positive and negative reviews:") 
print(df["Pos"].value_counts(normalize = True))

print("<---- Succesfully created bar charts and viewed them")
""" 
                                                ************************
                                                *Generating a wordcloud*
                                                ************************
"""

print("Generating a wordcloud ---->")

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

swords = ["based","need","could","range","offer","fairly","nights","attentiv","putney","fri","pool","f","booking","tube","com","Length","the","at","cl","i","dtype","object","o","Review","and","th","it","Name","our","dad",
"seating","we","shuttle","access","wit","or","of","ran","for","my","went","testing","was","after","to","very","but","with","without","late",
"grea","robes","no","king","slept","work","would","had","are","not","one","r","bed","a","all","plugs","though","t","on","previo","in","didn","junior","from","were","help","even"]
swords = swords + stopwords.words('english')

def show_wordcloud(data):
    wordcloud = WordCloud(
        stopwords = swords,
        background_color = 'white',
        max_words = 50,
        max_font_size = 40,
        scale = 3,
        random_state = 42,
        min_word_length = 3
    ).generate(str(data))

    plt.axis('off')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

dfPos = df[df['Pos'] == 1]
dfNeg = df[df['Pos'] == 0]

print("Pos and Neg combined:")
show_wordcloud(df["Review"])

print("Positive reviews:")
show_wordcloud(dfPos["Review"])

print("Negative reviews:")
show_wordcloud(dfNeg["Review"])

print("<---- Succesfully created a wordcloud and viewed it")

"""
*******************************************************************************************************************************************************************************************************************************************************************************************
                                                    *MODEL BUILDING*  
******************************************************************************************************************************************************************************************************************************************************************************************
"""

"""
                                                ***************
                                                *Preprocessing*  
                                                ***************
"""
print("Building three different models using supervised learning classifiers ---->")

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize

def tok(text):
    return word_tokenize(text)

vectorizer = TfidfVectorizer(min_df=20, max_df=0.95, ngram_range=(1,1), stop_words='english', tokenizer=tok,strip_accents='ascii',lowercase=True)

X = df["Review"]

y = df['Pos']

X=vectorizer.fit_transform(df["Review"])

print(X)

print("Shapes of X (Reviews-column) and y (Pos-column):")
print(y.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

print("Splitting/preprocessing done")

"""
                                                *************************
                                                *Multinomial Naive Bayes*  
                                                *************************
"""

nbModel = MultinomialNB()
nbModel.fit(X_train, y_train)
predictions = nbModel.predict(X_test)

outcome = pd.DataFrame(confusion_matrix(predictions, y_test), index = ['P: Good Review', 'P: Bad review'], columns = ['Actual: Good review', 'Actual: Bad review'])
print("Outcome of the confusion matrix of Naive Bayes stored into a dataframe: ")
print(outcome)
"""
                                                *********************
                                                *Logistic Regression*  
                                                *********************
"""
logregModel = LogisticRegression(C=5e25)
logregModel.fit(X_train,y_train)

predictions = logregModel.predict(X_test)

outcome = pd.DataFrame(confusion_matrix(predictions, y_test), index = ['P: Good Review', 'P: Bad review'], columns = ['Actual: Good review', 'Actual: Bad review'])
print("Outcome of the confusion matrix of Logistic Regression stored into a dataframe: ")
print(outcome)
"""
                                                        ******
                                                        *SVM*  
                                                        ******
"""
from sklearn.svm import SVC

dfLimited = df.head(2000)

dfLimitedX = dfLimited["Review"]
dfLimitedy = dfLimited["Pos"]

dfLimitedX = vectorizer.fit_transform(dfLimitedX)

lim_X_train, lim_X_test, lim_y_train, lim_y_test = train_test_split(dfLimitedX, dfLimitedy, test_size=0.2, random_state=0)

SVMmodel = SVC(kernel='linear')
SVMmodel.fit(lim_X_train, lim_y_train)
predictions = SVMmodel.predict(lim_X_test)

outcome = pd.DataFrame(confusion_matrix(predictions, lim_y_test), index = ['P: Good Review', 'P: Bad review'], columns = ['Actual: Good review', 'Actual: Bad review'])
print("Outcome of the confusion matrix of SVM stored into a dataframe: ")
print(outcome)


print("<---- Succesfully built three different models")
"""
                                                    **********
                                                    *Tweeking*  
                                                    **********
"""
from sklearn.pipeline import Pipeline
from sklearn.model_selection  import GridSearchCV
from sklearn.preprocessing import StandardScaler

sc = StandardScaler(with_mean=False)
pipeline = Pipeline(steps=[('sc', sc),('svc', SVMmodel)])


parameters =  [{
                    'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                    'svc__kernel': ['linear']
                  },
                 {
                    'svc__C': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                    'svc__gamma': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
                    'svc__kernel': ['rbf']
                 }]


grid = GridSearchCV(pipeline, param_grid=parameters, cv=5) 
grid.fit(lim_X_train, lim_y_train)

print ("Score, tweeking SVM = %3.2f" %(grid.score(lim_X_test,lim_y_test)))
print("Best parameters set found on development set:")
print (grid.best_params_)

grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_results.head()

"""
*************************************************************************************************************************************************************************************************************************************************************************************************
                                                        *MODEL EVALUATION/VIZ*  
***********************************************************************************************************************************************************************************************************************************************************************************************
"""

print("Evaluating the model ---->" )

print("roc_auc_score of Naive Bayes: ",roc_auc_score(y, nbModel.predict(X)))
print("roc_auc_score of Logistic Regression: ",roc_auc_score(y, logregModel.predict(X)))
print("roc_auc_score of SVM: ",roc_auc_score(dfLimitedy, SVMmodel.predict(dfLimitedX)))

print("<---- Succesfully evaluated the model and visualized the results")

"""
                                                    *************
                                                    *Pickle time*
                                                    *************
"""
print("Pickling the models ---->")

import pickle

fileNB = r"C:\Users\...
fileLogReg = r"C:\Users\..."
fileSVM = r"C:\Users\..."

pickle.dump(nbModel, open(fileNB, 'wb'))
pickle.dump(logregModel, open(fileLogReg, 'wb'))
pickle.dump(fileSVM, open(fileSVM, 'wb'))

print("Accuracy score of the Naive Bayes model = " + str(nbModel.score(X_test, y_test)))


saved_NB_Model = pickle.load(open(fileNB, 'rb'))

result = str(saved_NB_Model.score(X_test, y_test))

print("Accuracy score of Naive Bayes model from pickle =", result)

print("Outcome of the confusion matrix of Logistic Regression stored into a dataframe from pickle: ")
print(outcome)

print("<---- Succesfully compiled the script")
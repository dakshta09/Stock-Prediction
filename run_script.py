from flask import Flask, render_template
app = Flask(__name__)
import pprint
import google.generativeai as palm
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import openai
import os
from flask import Flask, render_template, request
#add your api key here
palm.configure(api_key='AIzaSyDqErL4GIRikBNqN6n3qggnR4mNMw8w6p4')
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = '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'
# We are working with Generate Text model only
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name

#function to call bard api and get prompt response
def palmresponse(prompt):
    completion = palm.generate_text(
    model=model,
    prompt=prompt,
    temperature=0,
    max_output_tokens=2000)

    return completion.result


def gpt_outcome_data(message):
    
    response = openai.Completion.create(
    engine='text-davinci-003',#"text-embedding-ada-002",  
    prompt=message,
    max_tokens=2000,
    n=1,
    stop=None)
    return response.choices[0].text.strip()

# function to convert the string output to pandas dataframe
def debtequitydf(bard):
    import sys
    if sys.version_info[0] < 3: 
        from StringIO import StringIO
    else:
        from io import StringIO

    TESTDATA = StringIO(bard)
    df = pd.read_csv(TESTDATA, sep="|")
    return df.iloc[1:,1:-1]


#function to generate prompt for different companies
def bardoutcome_data(company,type1):
    #add your prompt here
    prompt ="fetch the data for fields- total debt, total equity, total debt on total equity, diluted eps, total revenue,\
    net income, EBITDA,Long Term Investment,Interest Expense,Return on Equity,Cash \
    and Cash Equivalents ,Quick Ratio,Current Ratio  for "+company + "from FY 2020 to 2022 and  \
    triple checked this data and it is all consistent across\
    Google Finance, Yahoo Finance, and Investing.com."
    if type1=='Bard':
        response=palmresponse(prompt)
    elif type1=='ChatGpt':
        #print(prompt)
        response=gpt_outcome_data(prompt)
        #print(response)
    return response
# Also if "+company +" is a good company to invest in 2023 and \     how much % return someone can get on 1000 usd"


#function to generate prompt for different companies
def bardoutcome_investment(company, data,type1):
    #add your prompt here
    prompt ="Is "+company+" a good company to invest in 2023 and how much % return someone can expect to get on $1000. \
    Use this data to analyze - Total Debt,Total Equity,Total Debt on Total Equity,Diluted EPS,Total Revenue,Net Income,EBITDA,\
    Long Term Investment,Interest Expense,Return on Equity,Cash and Cash Equivalents ,Quick Ratio,Current Ratio  "+ data 
    
    if type1=='Bard':
        response=palmresponse(prompt)
    elif type1=='ChatGpt':
        response=gpt_outcome_data(prompt)
    return response

#function to generate prompt for different companies
def bardoutcome_investment1(company,data,type1):
    #add your prompt here
    prompt ="Is "+company+" a good company to invest in 2023 and how much % return someone can expect to get on $1000. \
    Use this data to analyze - Total Debt,Total Equity,Total Debt on Total Equity,Diluted EPS,Total Revenue,Net Income,EBITDA,\
    Long Term Investment,Interest Expense,Return on Equity,Cash and Cash Equivalents ,Quick Ratio,Current Ratio  "+ data[0] + data[1] + data[2] 
    
    if type1=='Bard':
        response=palmresponse(prompt)
    elif type1=='ChatGpt':
        response=gpt_outcome_data(prompt)
    return response

def sentiment_analysis(t1):
    prompt ="perform sentiment analysis of this text"+ str(t1) +"in single keyword as positive, negative or neutral"
    response = str(palmresponse(prompt))    
    if response == 'positive':
        return "Received " + response +" sentiment analysis from Bard",'Yes'
    elif response == 'negative':
        return "Received " + response +" sentiment analysis from Bard",'No'
    if response == 'neutral':
        return "Received " + response +" sentiment analysis from Bard",'Maybe'


#function to remove stopwords and generate bag of words
import nltk

def remove_stopwords(string):
    # Create a set of stopwords.
    stopwords = set(nltk.corpus.stopwords.words('english'))
    # Split the string into a list of words.
    #print(string)
    words = str(string).split()
    # Remove the stopwords from the list of words.
    filtered_words = [word for word in words if word not in stopwords]
    # Join the filtered words back into a string.
    filtered_string = ' '.join(filtered_words)
    # Return the filtered string.
    return filtered_string



def complete_process(company_name,dates,type1):
    
    #empty dataframe to store result
    final_data=pd.DataFrame(columns=['Company_name','dates','bard_company_info','bard_recommendation', "Investment_Y/N",    "Sentiment_of_investment", "Latest_news","sentiment_of_latest_news", "past_news", "highest_match", "sentiment_of_match"],index=range(15))
    
    final_data.iloc[:,0]=company_name
    final_data.iloc[:,1]=dates

    bard_data=bardoutcome_data(company_name,type1)
    #print(bard_data)
    bard_outcome=bardoutcome_investment(company_name, bard_data,type1)
    #print(bard_outcome)
    
    sentiment_bard, flag=sentiment_analysis(bard_outcome)
    #print(sentiment_bard)
    prompt ="latest wall street journal news for "+company_name +" in ** format"
    t1=palmresponse(prompt)
    #print(t1)
    #print("This is the latest news only headline in wall street journal for "+ company_name)
    
    t1=t1.split("**")[2]
    s1,flag1=sentiment_analysis(t1)
    #print(s1)
    
    from io import StringIO
    df = pd.read_csv(StringIO(palmresponse("pull " + company_name +" news from wall street journal only for these dates " + \
                                           dates +  " in tabular format")), sep="|")
    df=df.iloc[1:,1:-1]
    #print(df)
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=50)

    # Transform the texts into TF-IDF vectors.
    hist_vector = vectorizer.fit_transform(df.iloc[:,1].apply(lambda x: remove_stopwords(x)))
    #print(hist_vector)
    #print("hist_vector")
    current_vector=vectorizer.transform(pd.Series(t1).apply(lambda x: remove_stopwords(x)))
    #print(current_vector)
    from sklearn.metrics.pairwise import cosine_similarity
    cos=cosine_similarity( hist_vector,current_vector)
    index_highest=pd.DataFrame(cos).idxmax()[0]
    #print("sentiment analysis")
    s2=sentiment_analysis(df.iloc[index_highest,2])
    #print(index_highest)
    #print(s2)
    final_data.iloc[:,2:]= bard_data, bard_outcome, flag,sentiment_bard, t1,s1,df.iloc[:,1],df.iloc[index_highest,1],s2
    print(final_data)

    # load json and create model
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")


    dataset_test = pd.read_csv('GOOG.csv')
    actual_stock_price = pd.DataFrame(dataset_test).values
    dataset_total = pd.concat((dataset_test['Open'], actual_stock_price), axis = 0)
    inputs = dataset_train[-81:]['Open'].values

    inputs = inputs.reshape(-1,1)
    inputs = scaler.transform(inputs)

    x_test = []
    for i in range(60,80):
        x_test.append(inputs[i-60:i])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test , (x_test.shape[0], x_test.shape[1],1))
    x_test = np.reshape(x_test , (x_test.shape[0], x_test.shape[1],1))
    predicted_stock_price = regressor.predict(x_test)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    return final_data, predicted_stock_price



@app.route('file:///C:/Users/HP%20NOTES/Mini%202/exp%20learning/mini%202/UI%20code/UI.html')
def run_script(company_name, date, type1):
    print('hi')
    return complete_process(company_name,date,type1)

	
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,render_template,request
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



app=Flask(__name__)

@app.route("/")
def hello():
    return render_template('home.html')

@app.route('/output',methods=['post'])

def output():
    text1=request.form.get('data1')
    text2=request.form.get('data2')

    documents=[text1,text2]
    bow = CountVectorizer(stop_words = 'english')#we can also use tfidf
    sparse_mat = bow.fit_transform(documents)#we fit the count vector in matrix form
    df = pd.DataFrame(sparse_mat.toarray() , columns = bow.get_feature_names() , 
                  index = ['text1','text2'])

    print (cosine_similarity(df,df))
    answer=cosine_similarity(df,df)
    if answer[0][1]>0.8:
        simi='texts are similar'
    else :
        simi='not similar'   
    return render_template("home.html",predict= {simi}) 

app.run(debug=True)    
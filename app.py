# -*- coding: utf-8 -*-
"""
Created on Tue May 19 11:15:44 2020

@author: Suyog
"""

#Importing Libraries
import pandas as pd 
from flask import Flask,render_template,request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

# defining a function that recommends 10 most similar movies
def call_recommend(m):
    m = m.lower()
    movie = pd.read_csv('movie_data_final.csv')
    
    # check if the movie is in our database or not
    if m not in movie['original_title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        ## Content Based Recommendation System
        ### Using Tf-IDF Vectorizer to formulate vectorization matrix 
       
            
        tf = TfidfVectorizer(
               min_df=3,
               max_features=None,
               strip_accents='unicode',
               analyzer='word',
               token_pattern=r'\w{1,}',
               ngram_range=(1,3),
               stop_words='english'
        )
        
        #Fitting the TF-IDF on 'overview' text
        tf_matrix = tf.fit_transform(movie['overview'].values.astype('U'))
            
        #Compute sigmoid kernel
        sig = sigmoid_kernel(tf_matrix,tf_matrix)
            
        #Reverse mapping of indices and movie titles
        indices = pd.Series(movie.index,index=movie['original_title']).drop_duplicates()
        
        #Get the corresponding to original_title
        index = indices[m]
        
        #Get the pairwise similiarity scores
        sig_scores = list(enumerate(sig[index]))
        
        #Sort the movies
        sig_scores = sorted(sig_scores,key=lambda x :x[1],reverse=True)
        
        #Score of 10 most similar movies
        sig_scores = sig_scores[1:11]
        
        #Movie indices
        movie_indices = [i[0] for i in sig_scores]
        
        movieList = movie['original_title'].iloc[movie_indices]
        #movieList.columns = ['Movie Name','Rating']
        #movieList = movieList.sort_values(['Rating'],ascending=False)
        #Top 10 most similar movies
    return movieList

#testing cases
#finally top 10 list
#recommend = give_rec('The American')
#print(recommend)    
#input1 = input("Enter the movie name : ")
#recommend = give_rec(input1)
#print(recommend)
    

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = call_recommend(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s',type_r='type_r')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l',type_r='type_r')
    


if __name__ == '__main__':
    app.run(debug=True)
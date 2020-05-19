# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:08:19 2020

@author: Suyog
"""

#Importing Libraries
import pandas as pd 

#Pre-processing
#Importing Dataset, download in kaggle site : https://www.kaggle.com/tmdb/tmdb-movie-metadata
credit_data = pd.read_csv('tmdb_5000_credits.csv')
movie_data = pd.read_csv('tmdb_5000_movies.csv')

credit_col_rename = credit_data.rename(index=str,columns={'movie_id':'id'})

#Merge those two dataset using ID as foreign key
movie = movie_data.merge(credit_col_rename,on='id')

un_col=['homepage','title_x','title_y','status','production_companies']
movie[un_col]

#Lets drop the unnecessary columns
movie = movie.drop(columns=['homepage','title_x','title_y','status','production_companies'])

movie['overview']  = movie['overview'].fillna('')
movie['original_title'] = movie['original_title'].str.lower()

movie.to_csv('movie_data_final.csv',index=False)

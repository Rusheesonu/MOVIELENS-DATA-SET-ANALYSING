#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:54:00 2018

@author: rishii
"""
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
ratings_base.shape, ratings_test.shape

import graphlab
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(ratings_test)

popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating')


#Get recommendations for first 5 users and print them
#users = range(1,6) specifies user ID of first 5 users
#k=5 specifies top 5 recommendations to be given

popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)

ratings_base.groupby(by='movie_id')['rating'].mean().sort_values(ascending=False).head(20)

#Train Model


item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='movie_id', target='rating', similarity_type='pearson')

#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)


#Comparing both models

model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])
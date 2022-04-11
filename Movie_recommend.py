#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[58]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[59]:


#id is movie_id on tmdb website
movies.head(1)


# In[60]:


credits.head(1)


# In[61]:


credits['crew']


# In[62]:


movies=movies.merge(credits,on='title')


# In[63]:


movies.info()


# In[64]:


# genres,id,keywords,view,cast,crew
movies=movies[['id','genres','keywords','overview','cast','crew','title']]


# In[65]:


movies.head()


# In[66]:


movies.isna().sum()


# In[67]:


movies.dropna(inplace=True)


# In[68]:


movies.duplicated().sum()


# In[69]:


import ast


# In[70]:


def extract(obj):
    ls=[]
    for i in ast.literal_eval(obj):
        ls.append(i['name'])
    return ls


# In[71]:


movies['genres']=movies.genres.apply(extract)


# In[72]:


movies['keywords']=movies.keywords.apply(extract)


# In[73]:


movies.cast


# In[74]:


def extract(obj):
    ls=[]
    count=0
    for i in ast.literal_eval(obj):
        if count!=3:
            ls.append(i['name'])
            count+=1
        else:
            break;
    return ls


# In[75]:


movies['cast']=movies.cast.apply(extract)


# In[76]:


def extract_director(obj):
    ls=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director' or 'director':
            ls.append(i['name'])
            break
    return ls


# In[77]:


movies['crew']=movies.crew.apply(extract_director)


# In[78]:


movies.head()


# In[79]:


#converting overview column to list
movies['overview']=movies.overview.apply(lambda x:x.split())


# In[80]:


movies.head()


# In[81]:


movies['genres']=movies.genres.apply(lambda x:[i.replace(" ",'')for i in x])
movies['keywords']=movies.keywords.apply(lambda x:[i.replace(" ",'')for i in x])
movies['cast']=movies.cast.apply(lambda x:[i.replace(" ",'')for i in x])
movies['crew']=movies.crew.apply(lambda x:[i.replace(" ",'')for i in x])


# In[82]:


movies['tags']=movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[89]:


new_movies=movies[['id','title','tags']]


# In[91]:


#joining over spaces
new_movies['tags']=new_movies.tags.apply(lambda x:" ".join(x))


# In[92]:


new_movies.tags[0]


# In[145]:


new_movies.info()


# In[116]:


import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()


# In[117]:


def stem(text):
    l=[]
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)
        


# In[118]:


new_movies['tags']=new_movies['tags'].apply(stem)


# In[119]:


#vectorization
#converting texts to vectorization using bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[123]:


vectors=cv.fit_transform(new_movies['tags']).toarray()


# In[124]:


cv.get_feature_names()


# In[122]:


#calculationg cosine distance-angle between vectors representing movies
#smaller the angle,movies are more similar
from sklearn.metrics.pairwise import cosine_similarity


# In[130]:


similarity=cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x:x[1])[1:6]


# In[143]:


def recommend(movie):
    movie_index=new_movies[new_movies['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
    
    
    for i in movies_list:
        print(new_movies.iloc[i[0]].title)
    
    
    
    


# In[144]:


recommend('Avatar')


# In[146]:


import pickle

pickle.dump(new_movies,open('movie_recommender.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:
def main():
    import streamlit as st
    import requests

    st.title('Movie Recommendation System')

     #st.header('Find movies of your choice!!!!!')

    import pickle

    def fetch_poster(movie_id):
        url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
        data = requests.get(url)
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path

    def recommend(movie):
        index = movies[movies['title'] == movie].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movie_names = []
        recommended_movie_posters = []
        for i in distances[1:6]:
            # fetch the movie poster
            movie_id = movies.iloc[i[0]].movie_id
            recommended_movie_posters.append(fetch_poster(movie_id))
            recommended_movie_names.append(movies.iloc[i[0]].title)

        return recommended_movie_names,recommended_movie_posters


    st.header('Movie Recommender System')
    movies = pickle.load(open('movie_recommender.pkl','rb'))
    similarity = pickle.load(open('similarity.pkl','rb'))

    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list)

    if st.button('Show Recommendation'):
        recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
        col1, col2, col3, col4, col5 = st.beta_columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])

        with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])

if __name__=="__main__":
    main()

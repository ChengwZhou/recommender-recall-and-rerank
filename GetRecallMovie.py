import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import itertools
from collections import Counter

from MovieClass import MovieClass, MultiMovieClass
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def find_top_view_movies():
    #from full dataset
    df = pd.read_csv('data_raw/ratings.csv')
    top_view_movies = df.movieId.value_counts().index.values[:50]
    return top_view_movies

def get_user_history_embedding(model, movie_list):
    user_history_embedding = []
    for i in range(len(movie_list)):
        if model.wv.key_to_index[movie_list[i]]:
            movie_vector = model.wv[movie_list[i]]
            user_history_embedding.append(movie_vector)
    return np.array(user_history_embedding)

def get_movie_vector(model, movie_list):
    movie_v_list = []
    for i in range(len(movie_list)):
        movie_vector = model.wv[movie_list[i]]
        movie_v_list.append(movie_vector)
    return np.array(movie_v_list)


def get_user_movie_similarity(user_v, movie_vs):
    similarity = []
    for movie_v in movie_vs:
        siml = cosine_similarity(user_v.reshape(1, -1), movie_v.reshape(1, -1))
        similarity.append(siml[0][0])
    return similarity


def movie_popularity(movie_list, df_pop):
    pop_list = []
    for i, m_id in enumerate(movie_list):
        if not df_pop.loc[df_pop.movieId == m_id].empty:
            p = df_pop.loc[df_pop.movieId == m_id].populairty.values
        else:
            p = 1
        pop_list.append(p)
    return pop_list


def bulid_recall_movie_feature(df, df_pop, model_wv, top_20_movies, top_view_movies):
    movie_list = df.movieId.tolist()
    unique_movie = list(dict.fromkeys(movie_list))
    # gererate from movie2movie
    movies_2 = list(itertools.chain(*[top_20_movies[id] for id in unique_movie if id in top_20_movies]))
    recall_list1 = [[id, cnt] for id, cnt in Counter(movies_2).most_common(100) if id not in unique_movie]

    # add top_view_movies
    recall_id = [i[0] for i in recall_list1] + list(top_view_movies)
    recall_id = list(dict.fromkeys(recall_id))

    recall_cnt = [i[1] for i in recall_list1]
    mean_cnt = np.mean(np.array(recall_cnt))
    recall_cnt = [i[1] for i in recall_list1] + [mean_cnt] * (len(recall_id) - len(recall_cnt))

    # user vector = mean of user history embeding
    user_history_embedding = get_user_history_embedding(model_wv, movie_list)
    user_vector = np.mean(user_history_embedding, axis=0)
    user_variance = np.std(user_history_embedding, axis=0)

    # movie vector and similarity
    movie_vectors_list = get_movie_vector(model_wv, recall_id)
    # simil_l = get_user_movie_similarity(user_vector, movie_vectors_list)

    # movie popularity
    pop_list = movie_popularity(recall_id, df_pop)

    df_recall = pd.concat(
        [pd.DataFrame(recall_id), pd.DataFrame([user_vector] * len(recall_id)), pd.DataFrame(movie_vectors_list),
         pd.DataFrame([user_variance] * len(recall_id)), pd.DataFrame(pop_list)], axis=1)
    return df_recall


def get_label(df_recall, df_y):
    groudtruth = df_y.groupby('userId').movieId.apply(list).to_dict()
    label = []
    for i in range(len(df_recall)):
        movieid = int(df_recall.iloc[i].movieId)
        userid = int(df_recall.iloc[i].userId)
        if movieid in groudtruth[userid]:
            label.append(1)
        else:
            label.append(0)
    return np.array(label)

def df_covisitation_to_dict(df):
    return df.groupby('movieId_x').movieId_y.apply(list).to_dict()



if __name__ == "__main__":
    #load data
    df_cov = pd.read_csv('data/Movie2Movie.csv')
    df_X = pd.read_csv('data/users_history.csv')
    df_y = pd.read_csv('data/ground_truth.csv')
    # df_dc = pd.read_csv('data/movie_director_casts.csv')
    df_pop = pd.read_csv('data/popularity.csv')
    # model_wv = Word2Vec.load("word2vec.model")
    model_wv = Word2Vec.load("word2vec_movie.model")

    top_20_cov_movies = df_covisitation_to_dict(df_cov)
    top_view_movies = find_top_view_movies()

    df_recall = df_X.groupby('userId').apply(lambda x: bulid_recall_movie_feature(x, df_pop, model_wv, top_20_cov_movies, top_view_movies))
    df_recall = df_recall.reset_index('userId')
    # print(df_recall)
    df_recall.columns.values[1] = 'movieId'
    df_recall.to_csv('data/dataset_feature.csv')
    label = get_label(df_recall, df_y)
    df_x = df_recall.iloc[:, 2:]
    print(df_x.shape)
import numpy as np
import pandas as pd
import itertools

class MovieClass:
    def __init__(self, movieId = None, tmdbId = None):
        """
        One of these params must be provided.
        If tmdbID is provided, there is no need to search movieID
        :param movieId: int
        :param tmdbId: int
        """
        self.movieId = movieId
        self.imdbId = None
        self.tmdbId = tmdbId
        self.DirectorId = None
        self.DirectorName = None
        self.CastsList = None
        self.keywordId = None
        self.__get_movie()

    def __get_movie(self):
        #get imdbId, tmdbId
        if not self.tmdbId:
            df = pd.read_csv('data_raw/links_small.csv')
            self.imdbId = int(df.loc[df.movieId == self.movieId].values[0][1])
            self.tmdbId = int(df.loc[df.movieId == self.movieId].values[0][2])

        #get director
        df3 = pd.read_csv('data_raw/credits.csv')
        for member in eval(df3.loc[df3.id == self.tmdbId].values[0][1]):
            if member['department'] == 'Directing' and member['job'] == 'Director':
                self.DirectorId = member['id']
                self.DirectorName = member['name']

        #get casts
        list = []
        for member in eval(df3.loc[df3.id == self.tmdbId].values[0][0]):
            list.append(member['id'])
            # print(member)
        self.CastsList = list

        #get keyword id
        ky_list = []
        df1 = pd.read_csv('data_raw/keywords.csv')
        if len(df1.loc[df1.id == self.tmdbId].values):
            for key in eval(df1.loc[df1.id == self.tmdbId].values[0][1]):
                ky_list.append(key['id'])
        self.keywordId = ky_list
        #


    def get_keywords_info(self):
        df1 = pd.read_csv('data_raw/keywords.csv')
        if len(df1.loc[df1.id == self.tmdbId].values):
            return eval(df1.loc[df1.id == self.tmdbId].values[0][1])
        else:
            return -1

    def get_metadata_info(self):
        df2 = pd.read_csv('data_raw/movies_metadata.csv', low_memory=False)
        return df2.loc[df2.id == str(self.tmdbId)].values

    def get_cast_info(self):
        df3 = pd.read_csv('data_raw/credits.csv')
        return eval(df3.loc[df3.id == self.tmdbId].values[0][0])

    def get_crew_info(self):
        df3 = pd.read_csv('data_raw/credits.csv')
        return eval(df3.loc[df3.id == self.tmdbId].values[0][1])


class MultiMovieClass:
    def __init__(self, movieIdlist):
        """
        :param movieId: list
        """
        self.movieId_list = movieIdlist
        self.imdbId_list = None
        self.tmdbId_list = None
        self.__get_movies_id()

    def __get_movies_id(self):
        df = pd.read_csv('data_raw/links_small.csv')
        temp_list = []
        for i, id in enumerate(self.movieId_list):
            if not df.loc[df.movieId == id].isna().values.any():
                tmdbId = int(df.loc[df.movieId == id].values[0][2])
                temp_list.append(tmdbId)
            else:
                temp_list.append(-1)
        self.tmdbId_list = temp_list


    def get_keywords(self):
        df1 = pd.read_csv('data_raw/keywords.csv')
        temp_list = []
        for i, id in enumerate(self.tmdbId_list):
            key_list = []
            if len(df1.loc[df1.id == id].values):
                for key in eval(df1.loc[df1.id == id].values[0][1]):
                    key_list.append(key['name'])
            temp_list.append(key_list)
        return temp_list

    def get_Directors_Casts(self):
        #get director list
        df3 = pd.read_csv('data_raw/credits.csv')
        #Directors
        director_list = []
        for i, id in enumerate(self.tmdbId_list):
            for member in eval(df3.loc[df3.id == id].values[0][1]):
                if member['department'] == 'Directing' and member['job'] == 'Director':
                    director_list.append(member['id'])
        #Casts
        cast_list = []
        for i, id in enumerate(self.tmdbId_list):
            list = []
            for member in eval(df3.loc[df3.id == id].values[0][0]):
                list.append(member['id'])
            list = list[:10] if len(list) > 10 else list
            cast_list.append(list)
        return director_list, cast_list

    def get_movieid_from_tmdb_list(self, tmdb_list):
        df = pd.read_csv('data_raw/links.csv')
        temp_list = []
        for i, id in enumerate(tmdb_list):
            tmdbId = int(df.loc[df.tmdbId == id].values[0][0])
            temp_list.append(tmdbId)
        return temp_list



if __name__ == "__main__":
    movies = MultiMovieClass([1,2,3])
    # print(movies.tmdbId_list)
    # print(movies.get_keywords())
    print(movies.get_Directors_Casts())

    # movie = MovieClass(movieId=1)
    # # print(movie.tmdbId, movie.get_metadata_info())
    # # print(movie.get_cast_info())
    # # print(f'director:{movie.DirectorName}')
    # print(f'cast list:{movie.get_cast_info()}')
    # print(f'keyword:{movie.get_keywords_info()}')

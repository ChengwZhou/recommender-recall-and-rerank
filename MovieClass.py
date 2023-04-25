import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    movie = MovieClass(movieId = 1, tmdbId = None)
    print(movie.get_keywords_info())
    print(movie.tmdbId, movie.get_metadata_info())
    # print(movie.get_cast_info())
    # print(f'director:{movie.DirectorName}')
    # print(f'cast list:{movie.CastsList}')
    # print(f'keyword:{movie.keywordId}')

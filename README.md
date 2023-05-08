# EE541-project-MovieRecommendationSystem
Plese following instructions and steps liseted below to run the prediction model

### Step-1: data pre-processing
Please allocate all the raw dataset files into the directory: data_raw. 

Follow DatapreProcess.ipynb as instructions to save all the requiered processed dataset files and also a Word2Vec model, listing as follow:  

  data/users_history.csv # create new directory data. 

  data/ground_truth.csv 

  data/Movie2Movie.csv

  data/popularity.csv

  word2vec_movie.model

### Step-2: build feature and model
run GetRecallMovie.py and save feature datafile data/dataset_feature.csv

Finally, run MovieRecommendationMLP.py

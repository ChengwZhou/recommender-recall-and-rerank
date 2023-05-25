# MovieRecommendationSystem

## The goals of this project are as follows: 

1.Data preprocessing and feature engineering: Clean and prepare the dataset for analysis, identify relevant features, and potentially create new ones to improve model performance. 

2.Exploratory Data Analysis (EDA): Analyze the dataset to gain insights into player performance trends and relationships between various features. 

3.Build Recall System: get co-visitation matrix from the training set. Return recall movie as candidates. 

3.Building Word Embedding Model: train Word2Vec model and use it to conduct feature engineering and users and the candidate movies. 

3.Model selection and training: Select appropriate machine learning and deep learning algorithms, split the data into training and testing sets, and train the models to make predictions. For the deep learning approach, we will perform reranking on the 100 movies recalled for each user. Ultimately, we will recommend the top 20 ranked movies to the users. 

4.Model Evaluation and Comparison: Evaluate the performance of the trained models using appropriate metrics and compare their accuracy to determine the most suitable model for predicting players’ performance. This step also involves fine-tuning the models to optimize their accuracy, as needed. And to measure the accuracy of the model, we consider top 20 highest scored predictions as the output, and use the valuation function: 

<img width="822" alt="image" src="https://github.com/ChengwZhou/tmdbMovie_Recommendation_System/assets/131209977/fa3bd825-b9e6-4024-b598-a5e3a5bc64b4">

overview of recommendation system steps: 
 
<img width="770" alt="image" src="https://github.com/ChengwZhou/tmdbMovie_Recommendation_System/assets/131209977/04991a59-9674-4af8-a2d6-7e07c7d787e4">

## Dataset
These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset[2]. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages. This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.
Refer to the website for further dataset introduction: Kaggle.


## Plese following instructions and steps liseted below to run the prediction model 

### Step-1: data pre-processing
Please rellocate all the raw dataset files into the directory: data_raw. 

Follow DatapreProcess.ipynb as instructions to save all the requiered processed dataset files and also a Word2Vec model, listing as follow:  

  data/users_history.csv # create new directory data. 

  data/ground_truth.csv 

  data/Movie2Movie.csv

  data/popularity.csv

  word2vec_movie.model

### Step-2: build feature and model
run GetRecallMovie.py and save feature datafile data/dataset_feature.csv

Finally, run train_Resnet50.ipynb to train a ResNet50 model and then predict  

or run MovieRecommendationMLP.py to train a MLP model and then predict

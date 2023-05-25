# tmdbMovie Recommendation System

## The goals of this project are as follows: 

1. Data preprocessing and feature engineering: Clean and prepare the dataset for analysis, identify relevant features, and potentially create new ones to improve model performance. 

2. Exploratory Data Analysis (EDA): Analyze the dataset to gain insights into player performance trends and relationships between various features. 

3. Build Recall System: get co-visitation matrix from the training set. Return recall movie as candidates. 

4. Building Word Embedding Model: train Word2Vec model and use it to conduct feature engineering and users and the candidate movies. We used genism library to train Word2Vec model based on keyword of each movie. And we set up vector size as 32, window as 5 and training epoch as 30, then saved the model for future usage. In this way, we can compute movie vector by calculating the average of all the keywords of this movie, and also can get user’s historical embedding from calculating the average of all the movie vectors of individual’s rating history.

5. Feature Engineering and Dimensionality Adjustment: We put great attention to capture appropriate features before feeding data to Machine Learning model. In this task, we explored the word embedding and used it on our model. All details of feature engineering of our task are shown in Table below. we considered the feature from both user and candidate movie aspects. Therefore, we chose similarity of user and candidate calculating by cosine distance as the first feature. Then, user vector and variance of user vector as feature 2 and feature 3, to separately indicate the user’s interest zone’s center and the width of it. And added with movie vector and movie popularity as feature 3 and feature 5. Finally, we have 98 dimensions of feature for each recall movies in total. As we have so many dimensions of feature to considered when preform machine learning, we conducted PCA process to reduce the feature dimension to lower the degree. In this way, the dataset can be more separable and is capable to speed up the model processing.

<img width="795" alt="image" src="https://github.com/ChengwZhou/tmdbMovie_Recommendation_System/assets/131209977/aac15943-d5ad-48cd-9982-f7095585c4c0">



6. Model selection and training: Select appropriate machine learning and deep learning algorithms, split the data into training and testing sets, and train the models to make predictions. For the deep learning approach, we will perform reranking on the 100 movies recalled for each user. Ultimately, we will recommend the top 20 ranked movies to the users. 

7. Model Evaluation and Comparison: Evaluate the performance of the trained models using appropriate metrics and compare their accuracy to determine the most suitable model for predicting players’ performance. This step also involves fine-tuning the models to optimize their accuracy, as needed. And to measure the accuracy of the model, we consider top 20 highest scored predictions as the output, and use the valuation function: 

<img width="822" alt="image" src="https://github.com/ChengwZhou/tmdbMovie_Recommendation_System/assets/131209977/fa3bd825-b9e6-4024-b598-a5e3a5bc64b4">

overview of recommendation system steps: 
 
<img width="770" alt="image" src="https://github.com/ChengwZhou/tmdbMovie_Recommendation_System/assets/131209977/04991a59-9674-4af8-a2d6-7e07c7d787e4">

## Dataset
These files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages. This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.
Refer to the website for further dataset introduction: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings_small.csv


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

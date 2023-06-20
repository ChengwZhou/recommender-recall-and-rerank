# tmdbMovie Recommendation System

## Procedures and Contributions: 

* Data preprocessing and feature engineering: Clean and prepare the dataset for analysis, identify relevant features, and potentially create new ones to improve model performance. 

* Exploratory Data Analysis (EDA): Analyze the dataset to gain insights into player performance trends and relationships between various features. 

* Build Recall System: get co-visitation matrix from the training set. Return recall movie as candidates. 

* Building Word Embedding Model: train Word2Vec model and use it to conduct feature engineering and users and the candidate movies. We used ```genism``` library to train Word2Vec model based on keyword of each movie. And we set up ```vector_size``` as ```32```, ```window``` as ```5``` and ```epoch``` as ```30```, then saved the model for future usage. In this way, we can compute movie vector by calculating the average of all the keywords, and also can get user’s historical embedding from calculating the average of all the movie vectors of individual’s rating history.

* Feature Engineering: we considered the feature from both user and candidate movie aspects. The similarity of user and candidate calculating by cosine distance is the first feature. And user vector and variance of user vector are feature 2 and feature 3, to separately indicate the user’s interest zone’s center and the width of it, added with movie vector and movie popularity as feature 3 and feature 5. Totally, we have 98 dimensions of feature for each recall movies.

* Model selection and training: Select appropriate machine learning and deep learning algorithms, split the data into training and testing sets, and train the models to make predictions. For the deep learning approach, we will perform reranking on the 100 movies recalled for each user. Ultimately, we will recommend the top 20 ranked movies to the users. 

* Evaluation and Model Comparison: Evaluate the performance of the trained models using appropriate metrics and compare their accuracy to determine the most suitable model for prediction. This step also involves fine-tuning the models to optimize their accuracy, as needed. In experiments, we find CNN models(ResNet series) can reach best performance. And to measure the accuracy of the model, we consider top 20 highest scored predictions as the output, and use the valuation function: 

$$
P = \frac{|prediction \cap groundtruth|}{\min(20, |groundtruth|)}
$$

#### Overview: 
 
<img width="770" alt="image" src="https://github.com/ChengwZhou/tmdbMovie_Recommendation_System/assets/131209977/04991a59-9674-4af8-a2d6-7e07c7d787e4">

## Feature Engineering

 | Features | Dimension |
 |:--------:| :-------------:|
 | similarity between user and movie vector| 1 |
 | user vector  | 32 |
 | variance of user vector | 32 |
 | movie vector | 32 |
 | movie popularity  |  1 |

## Dataset
The dataset files contain metadata for all 45,000 movies listed in the Full MovieLens Dataset. The dataset consists of movies released on or before July 2017. Data points include cast, crew, plot keywords, budget, revenue, posters, release dates, languages, production companies, countries, TMDB vote counts and vote averages. This dataset also has files containing 26 million ratings from 270,000 users for all 45,000 movies. Ratings are on a scale of 1-5 and have been obtained from the official GroupLens website.
Refer to the website for further dataset introduction: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=ratings_small.csv


## Instructions

### Step-1: data pre-processing
Rellocate all the raw dataset files into the directory: ```data_raw```.

Follow ```DatapreProcess.ipynb``` as instructions to save all the requiered processed dataset files and also the trained Word2Vec model, listing as follow:  
```
data/users_history.csv # create new directory if not exist 

data/ground_truth.csv 

data/Movie2Movie.csv

data/popularity.csv

word2vec_movie.model
```

### Step-2: build feature and model
Get feature and save to ```data/dataset_feature.csv```.
```
python GetRecallMovie.py
```

Train a ResNet50 model and then predict (recommended),
```
python train_Resnet50.ipynb
```

or train a MLP model and then predict.
```
python MovieRecommendationMLP.py
```

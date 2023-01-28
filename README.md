# COVID-Vaccine-Stance-Detection
The idea is to do semantic analysis and detect category from Arabic Tweets. For every tweet , we should classify it as positive or negative or natural regrading to COVID219 Vaccine. The second task to do classification based on cateogry for same tweets.We have 7 types: Personal,News..etc 

>This serves as the project for the Natural Language Processing course taught to juniors in CUFE for 2023. We ranked 3 in our class
## Technologies 📚
<br>
<div align='center'>
<img src="https://github.com/Iten-No-404/COVID-Vaccine-Stance-Detection/blob/main/pytorch.png"  width="25%">
<img src="https://github.com/Iten-No-404/COVID-Vaccine-Stance-Detection/blob/main/nltk.png"  width="25%">
</div>
<br>


## Running the Application 🚀

1. Clone the repository
```
$ git clone https://github.com/Iten-No-404/COVID-Vaccine-Stance-Detection
```
# Pipeline 
## Preprocessing:
We defined our own tokenizer splitting on whitespaces & punctuations.
Cleaning tokens from punctuation.Removing tashkeel. Replacing variants of all arabic letters to their original parents (ؤ to و and أ to ا and so on).
Removing all extra Arabic characters (like المد). Remove any non-arabic charactersDefined our own set of stop words consisting of 2 & 3 characters (so as not to lose much semantics), then removed them.We used NLTK Stemmer to stem our output tokens.
We created 11 variants of our dataset, each with either Downsampling or Upsampling or both. Each with either equalised Stance Classes, or equalised Category Classes. All those Variants along with our Original Dataset without any up/downSampling.
## Feature Extraction:
###  Classical  features:
Sometimes , we used ***SMOTE*** to oversampling features to solve the problem of overfitting
* TF-IDF 
* N-gram
## Trainable features/embeddings:
* Word2Vec: CBOW using gensim models.
* Word2Vec: Skip-Gram using gensim models.
## Classical Classifiers:
* Multinomial Naive Bayes
* SVM Linear
* Random Forest
* Logistic Regression
* Model parameters are optimised using GridSearchCV like Random Forest
* 10-Fold Cross validation with random forest but it was overfitting
## Deep Learning models:
* Pytorch’s RNN 
* AraBert’s Transformer (Using transfer learning to train the last Fully Connected layer)
* XML Roberta (Using transfer learning to train the last Fully Connected layer)
* Qarib
* MarBert
## Final models:
For both the stance & category detection, we ended up using two arabert models trained on different variants of the dataset (Train 3 for stance & train 10 for category).
They were chosen because they gave the best F1-scores & especially the macro acreage scores. They ended up yielding nearly the best accuracy along with actually learning the dataset and predicting based on the tweet contents, not just predicting a single class & plainly overfitting.




# Best Results
## Arabert with stance 
Train data | f1-score      | accuracy | Macro avg |
|---------:|:-------------:|:--------:|----------:|
Train 3    |0.57,0.90,0.53 | 0.81     |0.66       |

## Arabert with category
Train data | accuracy | Macro avg |
|---------:|:--------:|----------:|
Train 10   | 0.588    | .438      |
Train 11   | 0.522    |0.392      |


# EXtra Results

### Features: TF-IDF Features

| Classifier    |  Naive Bayes  | Logistic Regression   | SVM linear   | Random Forest (100) | Random Forest(300) |
| ------------- |:-------------:| :--------------------:|:------------:|:-------------------:|-------------------:|
| stance        | 69.6          |   67.6                |  67.4        |  69.3               |   71.0             |
| category      |  56.6         |   63.08               |   65.0       |  65.3               |   64.5             |


Grid Search RandomForest for stance 0.67 accuracy , with f1-score :0.36,0.4,0.79
and  macro avg :0.52

## Features: Bigram

Classifier      |  Naive Bayes             | Logistic Regression            | SVM linear                     | Random Forest (100) 
| ------------- |:-----------------------:| :------------------------------:|:------------------------------:|-------------------:|
Stance F1-Score | 0.3,0.39,0.83           | 0.33,0.34,0.81                  | 0.32,0.36,0.81                 | 0.41,0.41,0.82     |
Stance Macro avg| 0.51                    | 0.49                            | 0.5                            |  0.52              |
CategoryMacroavg| 0.13                    | 0.31                            | 0.31                           | 0.23               |

## Features: TF-IDF and Bigram

Classifier      |  Naive Bayes            |  Logistic Regression            | SVM linear                     | Random Forest (100) | Random Forest (300)|  
| ------------- |:-----------------------:| :------------------------------:|:------------------------------:|:-------------------:|-------------------:|
Stance          |12.6                     | 80.4                            |12.6                            |80.4                 |84.0                |
Category        |54.5                     | 54.5                            |54.5                            |54.5                 |64.5                |


## Features : CBOW & Skip Gram & TF-IDF 

| Classifier    |  Naive Bayes  | Logistic Regression   | SVM linear   | Random Forest (100) | Random Forest(300) |
| ------------- |:-------------:| :--------------------:|:------------:|:-------------------:|-------------------:|
| stance        | 67.7          | 70.5                  | 12.7         | 71.0                | 76.8               |
| category      | 49.7          | 57.1996               | 13.9         | 12.8                | 54.555             |


## Arabert Word Embeddings with RNN
Task           | f1-score               |   Macro avg |
| ------------:|:----------------------:|------------:|
stance         |0.00,0.00,0.85          | 0.30        |
category       |0.71, the rest are 0.0  | 0.07        |







Catch Me If You Can [MLOps practice]
==============================

Web-user identification is a hot research topic on the brink of sequential pattern mining and behavioral psychology.

Here we try to identify a user on the Internet tracking his/her sequence of attended Web pages. The algorithm to be built will take a webpage session (a sequence of webpages attended consequently by the same person) and predict whether it belongs to Alice or somebody else.

The project was created for MLOps practice.

## Data Description
[Original data](https://www.kaggle.com/competitions/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/data).

The train set `train_sessions.csv` contains information on user browsing sessions where the features are:

 - `site_i` – are ids of sites in this session. The mapping is given with a pickled dictionary `site_dic.pkl`
 - `time_j` – are timestamps of attending the corresponding site
 - `target` – whether this session belongs to Alice


Project Organization
------------

    ├── README.md          
    ├── Docker              <- Docker files
    ├── cfg                 <- Configuration files
    │
    ├── data
    │   ├── external        <- Data from third party sources.
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   ├── raw             <- The original, immutable data dump.
    │   └── submission      <- Prediction data of the resulting model
    │
    ├── docs                <- A default Sphinx project
    ├── models              <- Trained and serialized models, model predictions, or model summaries
    ├── notebooks           <- Jupyter notebooks
    │
    ├── src                 <- Source code
    │   ├── api             <- Model Web Service Scripts
    │   │   ├── inference.py
    │   │   ├── lr_model.py
    │   │   ├── predicted_response.py
    │   │   └── inference.py
    │   │
    │   ├── config          <- Scripts to changing working settings
    │   │   └── config.py
    │   │
    │   ├── data            <- Scripts to data processing
    │   │   ├── add_missing_data.py
    │   │   ├── extract_target.py
    │   │   └── remove_bad_data.py
    │   │
    │   ├── features        <- Scripts to turn raw data into features for modeling
    │   │   ├── build_categorical_features.py
    │   │   ├── build_features_pipeline.py
    │   │   ├── build_numeric_features.py
    │   │   ├── build_preference_features.py
    │   │   ├── build_preferences_pipeline.py
    │   │   ├── build_union_pipeline.py
    │   │   ├── build_vectorizer.py
    │   │   └── build_vectorizer_pipeline.py
    │   │
    │   ├── models          <- Scripts to train models and then use trained models to make predictions
    │   │   ├── create_submission.py
    │   │   ├── optimize_model.py
    │   │   ├── train_model.py
    │   │   └── utils_mlflow.py
    │   │
    │   └── tests           <- Scripts for conducting tests
    │       └── test_raw_data.py
    │
    ├── .dvcignore          <- Dvcignore file
    ├── .env                <- Env file for Docker services, contains keys only for local services
    ├── .gitignore          <- Gitignore file
    │
    ├── docker-compose.yaml <- This is a Docker Compose file that will contain the instructions
    │                          necessary to start and configure the services
    │
    ├── dvc.yaml            <- DVC pipeline Settings
    │
    └── pyproject.toml      <- Standard project file, includes settings for additional components 
    

## Setup the project infrastructure 

1. Clone the repository to a local computer
```bash
git clone https://github.com/alexpetr0v/catch-me-if-you-can.git
```
2. Download poetry & create a virtual environment
```bash
pip install poetry
poetry install
```
3. Deploy Docker services
```bash
docker compose up -d --build
```
4. Run DVC pipeline
```bash
poetry run dvc repro
```
### To access the services:

 - mnnio s3 storage http://127.0.0.1:9001
 
 - mlflow server http://127.0.0.1:5000
 
 - model web-service http://127.0.0.1:6060
    
    
    
    
    
    
    

Catch Me If You Can [MLOps practice]
==============================

Web-user identification is a hot research topic on the brink of sequential pattern mining and behavioral psychology.

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

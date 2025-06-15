# Fake-Instagram-Account-Detection
This project is focused on detecting fake Instagram accounts using a combination of numerical and textual features. The pipeline includes data scraping, feature engineering, training multiple models, and finally combining them into a single ensemble model for better performance.

Dataset Setup
We used two datasets:

A small custom-scraped dataset that included usernames, bios, etc., mostly used for training NLP-based models.

A larger dataset used for training the numerical model on features like follower count, following count, number of posts, etc.

 Approach
We trained separate models on different types of data:

Textual Input Models (trained on small dataset):

BERT (fine-tuned) – F1: 0.988, Accuracy: 98.8%

TF-IDF + Classifier – F1: 0.84, ROC AUC: 0.95

BERT + Random Forest (Feature Extractor) – F1: 0.96, ROC AUC: 0.994

After comparison, we chose the best-performing NLP model to be part of the final ensemble.

Numerical Model (trained on large dataset): A Keras-based model using structured features like followers, posts, privacy status, etc.

Ensemble Model: Combined outputs of the numerical and NLP models using logistic regression as a meta-model.

Final Ensemble Performance (on held-out test set)
Accuracy: 96.5%

Precision: 94.7%

Recall: 98.6%

F1 Score: 96.6%

ROC AUC: 0.995

Deployment
You can run the final system using either:

A real Instagram username (which will be scraped)

Manually entered profile data (for testing fake-looking accounts)

The prediction output is binary (Fake/Real) with a confidence score.

Required Files
Before running the deployment script, make sure you have the following files in the same directory:

keras_model_wrapper1.pkl (Numerical model)

bert_model.pkl (Text-based model)

numerical_scaler.pkl (For scaling numerical features)

meta_model.pkl (Logistic regression ensemble model)

How to Use
Install required packages (instaloader, pandas, joblib, torch, etc.)

Run the deployment script (ensemble.ipynb or equivalent .py file).

Enter either Instagram usernames or manually crafted profile inputs.

The script will handle scraping, preprocessing, feature extraction, and prediction.

The Instagram scraping tool is rate-limited. To prevent being blocked, we’ve included a wait/sleep logic in case Instagram starts throttling.

If the username doesn’t exist, the script handles the error and skips that user.

Feature engineering steps (like digit ratio, bio length) are automatically applied during prediction time.


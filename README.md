# Network Security Modeling using Machine Learning

## Project Overview
This project analyzes and models network security using machine learning techniques, based on the UNSW-NB15 dataset, a well-known dataset for network traffic analysis and intrusion detection. The aim is to predict network intrusions or security events using advanced machine learning models by leveraging the network traffic features provided in the dataset.

## Models Implemented
- **Random Forest Classifier**: A classical machine learning model for initial predictions.
- **Sequential Neural Network**: A deep learning model built using TensorFlow/Keras to enhance prediction accuracy.

Both models are compared in terms of performance metrics such as accuracy, precision, recall, F1-Score, and training time.

## Requirements
Install the following Python libraries to run this project:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn plotly colorama pyfiglet joblib

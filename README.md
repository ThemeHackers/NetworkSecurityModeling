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
```
## Key Libraries
- **Pandas**: For data manipulation and cleaning.
- **Numpy**: For numerical operations.
- **Scikit-learn**: For machine learning models, preprocessing, and feature selection.
- **TensorFlow**: For building and training the neural network.
- **Seaborn & Matplotlib**: For generating static visualizations.
- **Plotly**: For interactive data visualizations.
- **Colorama & PyFiglet**: For enhanced terminal output.
## File Structure
Key output files generated by the project include:
- NetworkSecurityModeling.keras: The trained Sequential Neural Network model.
- confusion_matrix.png: A heatmap of the confusion matrix generated by the Random Forest model.
- Accuracy-Loss.png: Plots of training accuracy and loss for the Sequential Neural Network
## Project Workflow
Data Loading & Preprocessing
  - The dataset is loaded from a CSV file (UNSW-NB15) into a Pandas DataFrame.
  - Irrelevant columns such as id and attack_cat are removed.
  - Outliers in numeric columns are capped at the 95th percentile, and skewed distributions are normalized using logarithmic transformations.
  - Categorical columns are grouped for uncommon categories and encoded using OneHotEncoder.
    
Feature Selection
  - Chi-square (chi²) tests are used to rank feature importance.
  - The top 20 features are selected for model training, with their importance visualized using Plotly.

Modeling: Random Forest Classifier
  - The dataset is split into 80% training and 20% testing sets.
  - Numeric features are scaled using StandardScaler.
  - A Random Forest Classifier with 100 trees is trained. Performance is measured using:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
  - The confusion matrix is visualized as a heatmap and saved as confusion_matrix.png.

Modeling: Sequential Neural Network
  - A Sequential Neural Network is built using Keras, with dense layers of 128, 64, 32, and 16 units, each using ReLU activation.
  - The final layer uses a softmax activation for classification.
  - The model is compiled with the Adam optimizer and sparse categorical crossentropy loss function.
  - Training is conducted over 100 epochs with a batch size of 64. Training progress is visualized and saved as Accuracy-Loss.png.
  - The trained model is saved as NetworkSecurityModeling.keras for future use.

Performance Evaluation
  - After training both models, the following performance metrics are calculated:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
  - Training and prediction times
  - Metrics are printed in color-coded format using Colorama for better readability.

Saving the Model
  - The trained Sequential Neural Network is saved to disk using Keras as NetworkSecurityModeling.keras.




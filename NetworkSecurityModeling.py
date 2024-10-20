import json
import random
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.core.display import display, HTML
from colorama import Fore, Style
import pyfiglet
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from importlib import reload
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
import colorama
import kagglehub

path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")

print("Path to dataset files:", path)
colorama.init(autoreset=True)

def rainbow_text(text):
    colors = [Fore.RED, Fore.YELLOW, Fore.GREEN, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]
    rainbow_str = ""
    for char in text:
        rainbow_str += random.choice(colors) + char
    return rainbow_str + Style.RESET_ALL
ascii_art = pyfiglet.figlet_format("NetworkModeling", width=100)
print(rainbow_text(ascii_art))
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)
display(HTML("<style>div.output_scroll { height: 35em; }</style>"))
reload(plt)
plt.rcParams['figure.dpi'] = 200
warnings.filterwarnings('ignore')
pio.renderers.default = 'vscode'
pio.templates["ck_template"] = go.layout.Template(
    layout_colorway=px.colors.sequential.Viridis,
    layout_autosize=False,
    layout_width=800,
    layout_height=600,
    layout_font=dict(family="Calibri Light"),
    layout_title_font=dict(family="Calibri"),
    layout_hoverlabel_font=dict(family="Calibri Light"),
)
pio.templates.default = 'ck_template+gridon'
unsw_ = path + r'\\UNSW_NB15_training-set.csv'
print(Fore.BLUE + f'Dataset Path')
print(unsw_)
df = pd.read_csv(unsw_)
df.info()
df.head(10)
df.describe(include='all')
list_drop = ['id', 'attack_cat']
df.drop(list_drop, axis=1, inplace=True)
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('max = ' + str(df_numeric[feature].max()))
        print('75th = ' + str(df_numeric[feature].quantile(0.95)))
        print('median = ' + str(df_numeric[feature].median()))
        print(df_numeric[feature].max() > 10 * df_numeric[feature].median())
        print('----------------------------------------------------')
    if df_numeric[feature].max() > 10 * df_numeric[feature].median() and df_numeric[feature].max() > 10:
        df[feature] = np.where(df[feature] < df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))
        df_numeric = df.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
df_numeric = df.select_dtypes(include=[np.number])
df_before = df_numeric.copy()
DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = ' + str(df_numeric[feature].nunique()))
        print(df_numeric[feature].nunique() > 50)
        print('----------------------------------------------------')
    if df_numeric[feature].nunique() > 50:
        if df_numeric[feature].min() == 0:
            df[feature] = np.log(df[feature] + 1)
        else:
            df[feature] = np.log(df[feature])
df_numeric = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
DEBUG = 0
for feature in df_cat.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = ' + str(df_cat[feature].nunique()))
        print(df_cat[feature].nunique() > 6)
        print(sum(df[feature].isin(df[feature].value_counts().head().index)))
        print('----------------------------------------------------')
    if df_cat[feature].nunique() > 6:
        df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')
df_cat = df.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
X = df.iloc[:, 4:-2] 
y = df.iloc[:, -1]  
best_features = SelectKBest(score_func=chi2, k='all')
fit = best_features.fit(X, y)
df_scores = pd.DataFrame(fit.scores_)
df_col = pd.DataFrame(X.columns)
feature_score = pd.concat([df_col, df_scores], axis=1)
feature_score.columns = ['feature', 'score']
feature_score.sort_values(by=['score'], ascending=True, inplace=True)
fig = go.Figure(go.Bar(
    x=feature_score['score'][0:21],
    y=feature_score['feature'][0:21],
    orientation='h'))
fig.update_layout(title="Top 20 Features",
                  height=1200,
                  showlegend=False)
fig.show()

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
df.dropna(inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X.head()
feature_names = list(X.columns)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
for label in list(df_cat['state'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)
for label in list(df_cat['service'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)
for label in list(df_cat['proto'].value_counts().index)[::-1][1:]:
    feature_names.insert(0, label)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=y)
sc = StandardScaler()
X_train[:, 18:] = sc.fit_transform(X_train[:, 18:])
X_test[:, 18:] = sc.transform(X_test[:, 18:])
model_performance = pd.DataFrame(columns=['Accuracy', 'Recall', 'Precision', 'F1-Score', 'time to train', 'time to predict', 'total time'])
start_time = time.time()
sum([i ** 2 for i in range(1000000)])
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
start = time.time()
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0, bootstrap=True).fit(X_train, y_train)
end_train = time.time()
y_predictions = model.predict(X_test)
end_predict = time.time()
accuracy = accuracy_score(y_test, y_predictions)
recall = recall_score(y_test, y_predictions, average='weighted')
precision = precision_score(y_test, y_predictions, average='weighted')
f1s = f1_score(y_test, y_predictions, average='weighted')
print("Accuracy: " + "{:.2%}".format(accuracy))
print("Recall: " + "{:.2%}".format(recall))
print("Precision: " + "{:.2%}".format(precision))
print("F1-Score: " + "{:.2%}".format(f1s))
print("time to train: " + "{:.2f}".format(end_train - start) + " s")
print("time to predict: " + "{:.2f}".format(end_predict - end_train) + " s")
print("total: " + "{:.2f}".format(end_predict - start) + " s")
cm = confusion_matrix(y_test, y_predictions, labels=model.classes_)
model_performance.loc['Random Forest'] = [accuracy, recall, precision, f1s, end_train - start, end_predict - end_train, end_predict - start]
plt.rcParams['figure.figsize'] = 5, 5
sns.set_style("white")
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
model_performance.fillna(.90, inplace=True)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],))) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))  
model.add(Dense(16, activation='relu')) 
model.add(Dense(len(np.unique(y)), activation='softmax'))  
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
start_train_time = time.time()
history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=2)
end_train_time = time.time()

start_predict_time = time.time()
y_predictions = model.predict(X_test)
end_predict_time = time.time()

y_predictions_classes = np.argmax(y_predictions, axis=1)

accuracy = accuracy_score(y_test, y_predictions_classes)
recall = recall_score(y_test, y_predictions_classes, average='weighted')
precision = precision_score(y_test, y_predictions_classes, average='weighted')
f1s = f1_score(y_test, y_predictions_classes, average='weighted')

time_to_train = end_train_time - start_train_time
time_to_predict = end_predict_time - start_predict_time
total_time = time_to_train + time_to_predict

print(Fore.BLUE + "Accuracy: " + "{:.2%}".format(accuracy))
print(Fore.RED + "Recall: " + "{:.2%}".format(recall))
print(Fore.CYAN + "Precision: " + "{:.2%}".format(precision))
print(Fore.BLUE + "F1-Score: " + "{:.2%}".format(f1s))
print(Fore.WHITE + "time to train: " + "{:.2f}".format(time_to_train) + " s")
print(Fore.RED + "time to predict: " + "{:.2f}".format(time_to_predict) + " s")
print(Fore.YELLOW + "total: " + "{:.2f}".format(total_time) + " s")
acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy', color='tab:blue')
ax1.plot(epochs, acc, 'bo-', label='Training Accuracy')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', color='tab:red')
ax2.plot(epochs, loss, 'ro-', label='Training Loss')
ax2.tick_params(axis='y', labelcolor='tab:red')
plt.title('Training Accuracy and Loss')
fig.tight_layout()
plt.savefig('Accuracy-Loss.png')
model.save('NetworkSecurityModeling.keras')
print(Fore.GREEN + "Model saved as NetworkSecurityModeling.keras")

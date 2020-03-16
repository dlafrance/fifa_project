# Library import
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.makedirs('plots', exist_ok=True)

# Load file
df = pd.read_csv('fifa_20.csv', header=0)

# Drop positions that will not be comparable for analysis
df = df[(df.team_position != 'SUB') & (df.team_position != 'RES')]
print(df.team_position.value_counts())
print(df.shape)
print(df.columns.values.tolist())

df = df[['overall', 'team_position', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
         'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
         'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
         'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
         'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
         'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
         'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving',
         'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']]


# Reduce number of positions
def simple_position(df):
    if ((df['team_position'] == 'RB') | (df['team_position'] == 'LB') | (df['team_position'] == 'CB') |
            (df['team_position'] == 'LCB') | (df['team_position'] == 'RCB') | (df['team_position'] == 'RWB') |
            (df['team_position'] == 'LWB')):
        return 'DEF'
    elif ((df['team_position'] == 'LDM') | (df['team_position'] == 'CDM') | (df['team_position'] == 'RDM')):
        return 'DEF'
    elif ((df['team_position'] == 'LM') | (df['team_position'] == 'LCM') | (df['team_position'] == 'CM') |
          (df['team_position'] == 'RCM') | (df['team_position'] == 'RM')):
        return 'MF'
    elif ((df['team_position'] == 'LAM') | (df['team_position'] == 'CAM') | (df['team_position'] == 'RAM') |
          (df['team_position'] == 'LW') | (df['team_position'] == 'RW')):
        return 'MF'
    elif ((df['team_position'] == 'RS') | (df['team_position'] == 'ST') | (df['team_position'] == 'LS') |
          (df['team_position'] == 'CF') | (df['team_position'] == 'LF') | (df['team_position'] == 'RF')):
        return 'FWD'
    else:
        return df.team_position


df['position'] = df.apply(simple_position, axis=1)

# Drop both original fields and remove NAs
df = df.drop(['team_position'], axis=1)

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())
print(df.shape)

# linear regression
X = df.drop(['position'], axis=1)
y = df['position']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)
scaled_features = scaler.transform(X)

# Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, test_size=0.3)

# Printing original Dataset
print(f"X.shape: {X.shape}, y.shape: {y.shape}")

# Printing splitted datasets
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

# Training a Linear Regression model with fit()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score

print('Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix (diagonal is true)
print('Confusion Matrix')
print(confusion_matrix(y_test, predicted_values))

print('Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))

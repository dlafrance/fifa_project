# Library import
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.makedirs('plots', exist_ok=True)

# Load file and explore
df = pd.read_csv('fifa_20.csv', header=0)
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())


# Drop positions that will not be comparable for analysis
print(df.columns.values.tolist())

df = df[['overall', 'age', 'height_cm', 'weight_kg', 'player_positions', 'team_position', 'attacking_crossing',
         'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
         'skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
         'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
         'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
         'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
         'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
         'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving',
         'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']]

# Filling player's position for subs and reserves, as well as NAs from their positions selection
print(df.isnull().sum())
df['player_positions'] = df['player_positions'].str.split(',').str[0]

df['team_position'] = np.where((df['team_position'] == 'SUB'), df['player_positions'],
                               df['team_position'])
df['team_position'] = np.where((df['team_position'] == 'RES'), df['player_positions'],
                               df['team_position'])

df['team_position'].fillna(df['player_positions'], inplace=True)


# Reduce number of positions
def simple_position(df):
    if ((df['team_position'] == 'RB') | (df['team_position'] == 'LB') | (df['team_position'] == 'CB') |
            (df['team_position'] == 'LCB') | (df['team_position'] == 'RCB') | (df['team_position'] == 'RWB') |
            (df['team_position'] == 'LWB')):
        return 'DEF'
    elif (df['team_position'] == 'LDM') | (df['team_position'] == 'CDM') | (df['team_position'] == 'RDM'):
        return 'MID'
    elif ((df['team_position'] == 'LM') | (df['team_position'] == 'LCM') | (df['team_position'] == 'CM') |
          (df['team_position'] == 'RCM') | (df['team_position'] == 'RM')):
        return 'MID'
    elif ((df['team_position'] == 'LAM') | (df['team_position'] == 'CAM') | (df['team_position'] == 'RAM') |
          (df['team_position'] == 'LW') | (df['team_position'] == 'RW')):
        return 'MID'
    elif ((df['team_position'] == 'RS') | (df['team_position'] == 'ST') | (df['team_position'] == 'LS') |
          (df['team_position'] == 'CF') | (df['team_position'] == 'LF') | (df['team_position'] == 'RF')):
        return 'FWD'
    else:
        return df.team_position


df['position'] = df.apply(simple_position, axis=1)

# # Drop both original fields
df = df.drop(['player_positions', 'team_position'], axis=1)
print(df.isnull().sum())
print(df.position.value_counts())
print(df.shape)

# Box plot
sns.set(style="dark", palette="GnBu_d", color_codes=True)
position_order = ['FWD', 'MID', 'DEF', 'GK']
sns.boxplot(x='position', y='overall', order=position_order, data=df)
plt.title('Position and FIFA Rating')
plt.xlabel('Position')
plt.ylabel('FIFA Rating')
plt.savefig('plots/box_position_rating.png')
plt.clf()

# Distribution of skill rating by position
f, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
f.suptitle('Distribution of dribbling skill by position')
sns.set(style="dark", palette="GnBu_d", color_codes=True)
sns.distplot(df['skill_dribbling'][df['position'] == 'FWD'], ax=axes[0, 0])
axes[0][0].set(xlabel='FWD', ylabel='Frequency share')
sns.distplot(df['skill_dribbling'][df['position'] == 'MID'], ax=axes[0, 1])
axes[0][1].set(xlabel='MID')
sns.distplot(df['skill_dribbling'][df['position'] == 'DEF'], ax=axes[1, 0])
axes[1][0].set(xlabel='DEF', ylabel='Frequency share')
sns.distplot(df['skill_dribbling'][df['position'] == 'GK'], ax=axes[1, 1])
axes[1][1].set(xlabel='GK')
plt.savefig('plots/hist_rating_position.png')
plt.clf()

# Find best skills by position
player_features = ['attacking_crossing',
                   'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys',
                   'skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
                   'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
                   'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
                   'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression',
                   'mentality_interceptions',
                   'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
                   'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving',
                   'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']

for i, val in df.groupby(df['position'])[player_features].mean().iterrows():
    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))

# Logistic regression
X = df.drop(['position', 'overall'], axis=1)
y = df['position']

from sklearn.model_selection import train_test_split

# Splitting features and target datasets into: train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Printing original Dataset
print(f"X.shape: {X.shape}, y.shape: {y.shape}")

# Printing splitted datasets
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

# Training a LogisticRegression model with fit()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)

# Predicting the results for our test dataset
predicted_values = lr.predict(X_test)

# Printing accuracy score(mean accuracy) from 0 - 1
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# Printing the classification report
from sklearn.metrics import classification_report, confusion_matrix, f1_score

print('Classification Report')
print(classification_report(y_test, predicted_values))

# Printing the classification confusion matrix
print('Confusion Matrix')
cm = confusion_matrix(y_test, predicted_values)
print(confusion_matrix(y_test, predicted_values))

f, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap=plt.get_cmap('Blues'), ax=ax)

ax.set_xlabel('Predicted position')
ax.set_ylabel('True position')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['DEF', 'FWD', 'GK', 'MID'])
ax.yaxis.set_ticklabels(['DEF', 'FWD', 'GK', 'MID'])
plt.savefig('plots/confusion_matrix.png')
plt.clf()

print('Overall f1-score')
print(f1_score(y_test, predicted_values, average="macro"))

print('Coefficients')
print(pd.DataFrame(lr.coef_, columns=X.columns).to_string())
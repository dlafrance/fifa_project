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
df = df[(df.team_position != 'GK') & (df.team_position != 'SUB') & (df.team_position != 'RES')]
print(df.team_position.value_counts())
print(df.shape)
print(df.columns.values.tolist())

df = df[['age', 'height_cm', 'weight_kg', 'overall', 'wage_eur', 'preferred_foot',
         'international_reputation', 'weak_foot', 'skill_moves', 'team_position',
         'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
         'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
         'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
         'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',
         'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
         'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions',
         'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
         'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle']]


# Make preferred foot binary
def right_footed(df):
    if (df['preferred_foot'] == 'Right'):
        return 1
    else:
        return 0


df['right_foot'] = df.apply(right_footed, axis=1)


# Reduce number of positions
def simple_position(df):
    if ((df['team_position'] == 'RB') | (df['team_position'] == 'LB') | (df['team_position'] == 'CB') |
            (df['team_position'] == 'LCB') | (df['team_position'] == 'RCB') | (df['team_position'] == 'RWB') |
            (df['team_position'] == 'LWB')):
        return 'DF'
    elif ((df['team_position'] == 'LDM') | (df['team_position'] == 'CDM') | (df['team_position'] == 'RDM')):
        return 'DM'
    elif ((df['team_position'] == 'LM') | (df['team_position'] == 'LCM') | (df['team_position'] == 'CM') |
          (df['team_position'] == 'RCM') | (df['team_position'] == 'RM')):
        return 'MF'
    elif ((df['team_position'] == 'LAM') | (df['team_position'] == 'CAM') | (df['team_position'] == 'RAM') |
          (df['team_position'] == 'LW') | (df['team_position'] == 'RW')):
        return 'AM'
    elif ((df['team_position'] == 'RS') | (df['team_position'] == 'ST') | (df['team_position'] == 'LS') |
          (df['team_position'] == 'CF') | (df['team_position'] == 'LF') | (df['team_position'] == 'RF')):
        return 'ST'
    else:
        return df.team_position


df['position'] = df.apply(simple_position, axis=1)

# Drop both original fields and remove NAs
df = df.drop(['preferred_foot', 'team_position'], axis=1)

print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())
print(df.shape)

# linear regression
X = df.drop(['overall', 'wage_eur', 'position'], axis=1)
y = df['overall']

print(f'Dataset X shape: {X.shape}')
print(f'Dataset y shape: {y.shape}')

# Splitting features and target datasets into: train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Training a Linear Regression model with fit()
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# Output of the training is a model: a + b*X0 + c*X1 + d*X2 ...
columns_names = X.columns
print(f"Intercept: {lm.intercept_}\n")
print(f"Coeficients: {lm.coef_}\n")
print(f"Named Coeficients: {pd.DataFrame(lm.coef_, columns_names)}")

# Predicting the results for our test dataset
predicted_values = lm.predict(X_test)

# Printing the residuals: difference between real and predicted
for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")


sns.set(palette="inferno")

# Plotting differenct between real and predicted values
sns.scatterplot(y_test, predicted_values)
plt.plot([0, 50], [0, 50], '--')
plt.xlabel('Real Value')
plt.ylabel('Predicted Value')
plt.show()


# Plotting the residuals: the error between the real and predicted values
residuals = y_test - predicted_values
sns.scatterplot(y_test, residuals)
plt.plot([50, 0], [0, 0], '--')
plt.xlabel('Real Value')
plt.ylabel('Residual (difference)')
plt.show()

sns.distplot(residuals, bins=20, kde=False)
plt.plot([0, 0], [50, 0], '--')
plt.title('Residual (difference) Distribution')
plt.show()

from sklearn import metrics
print(f"Printing MAE error(avg abs residual): {metrics.mean_absolute_error(y_test, predicted_values)}")
print(f"Printing MSE error: {metrics.mean_squared_error(y_test, predicted_values)}")
print(f"Printing RMSE error: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
print(f"R2 Score: {metrics.r2_score(y_test, predicted_values)}")

sns.distplot(df['wage_eur'], bins=10, kde=False)
plt.title('Residual (difference) Distribution')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso

for Model in [LinearRegression, GradientBoostingRegressor, ElasticNet, KNeighborsRegressor, Lasso]:
    model = Model()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    print(f"Printing RMSE error for {Model}: {np.sqrt(metrics.mean_squared_error(y_test, predicted_values))}")
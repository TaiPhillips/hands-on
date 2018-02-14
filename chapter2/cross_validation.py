import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

"""
This is a simplified version of what has been explained in
Chapter 2. We dropped `ocean_proximity' which simplified
data manipulation a lot. Also, as there are no
hyperparameters to tweak we declared our linear regression
model the final candidate. After having run a
cross-validation we checked the performance of the model
against the test set. We used the RMSE again.
"""

housing = pd.read_csv('./housing.csv')
housing.drop('ocean_proximity', axis=1, inplace=True)
housing['income_class'] = np.ceil(housing['median_income'])
housing.where(housing['income_class'] < 5., 5., inplace=True)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
train_ixs, test_ixs = next(splitter.split(housing, housing['income_class']))
housing.drop('income_class', axis=1, inplace=True)

strat_train_set, strat_test_set = housing.iloc[train_ixs], housing.iloc[test_ixs]

train_set = strat_train_set.drop('median_house_value', axis=1)
train_lab = strat_train_set['median_house_value']

pipe = Pipeline([
	("fill_na", Imputer(strategy='median')),
	('feature_scale', StandardScaler()),
])

piped_train_set = pipe.fit_transform(train_set)

lreg = LinearRegression()
lreg.fit(piped_train_set, train_lab)

cv_scores = np.sqrt(-cross_val_score(lreg, piped_train_set, train_lab,
					scoring='neg_mean_squared_error', cv=10))

test_set = strat_test_set.drop('median_house_value', axis=1)
test_lab = strat_test_set['median_house_value']
piped_test_set = pipe.transform(test_set)

pred_test_lab = lreg.predict(piped_test_set)
rmse_lreg = np.sqrt(mse(pred_test_lab, test_lab))

print(
	f'*** CV estimates a score of ≈{np.mean(cv_scores)}\n'
	f'*** Score against test set {rmse_lreg}'
)

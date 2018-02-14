import pandas as pd
import numpy as np
import scipy.stats
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

housing = pd.read_csv('./housing.csv')

housing['income_class'] = np.ceil(housing['median_income'])
housing['income_class'].where(housing['income_class'] < 5., 5., inplace=True)

splitter = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
#                                                         _.^ Make execution reproducible
train_set_ix, test_set_ix = next(splitter.split(housing, housing['income_class']))

train_set = housing.iloc[train_set_ix].drop(['median_house_value', 'income_class'], axis=1)
train_lab = housing.iloc[train_set_ix]['median_house_value']

test_set = housing.iloc[test_set_ix].drop(['median_house_value', 'income_class'], axis=1)
test_lab = housing.iloc[test_set_ix]['median_house_value']

class DfSelector:
	def __init__(self, columns):
		self.columns = columns

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# pd.DataFrame is idempotent, luckily
		return X[self.columns]

class Factorizer:
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X.factorize()[0].reshape(-1, 1)

class OceanPromixitySimplifier:
	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X == '<1H OCEAN'

numpipe = Pipeline([
	('selectnums', DfSelector(train_set.drop('ocean_proximity',axis=1).columns)),
	('impute', Imputer(strategy='median')),
	('stdscale', StandardScaler()),
])

catpipe = Pipeline([
	('selectcats', DfSelector('ocean_proximity')),
	('rm_oceanpxs', OceanPromixitySimplifier()),
	('factorize', Factorizer()),
	('onehotencode', OneHotEncoder())
])

pipe = FeatureUnion([
	('numpipe', numpipe),
	('catpipe', catpipe)
])

piped_train_set = pipe.fit_transform(train_set)

linreg = LinearRegression()
linreg.fit(piped_train_set, train_lab)

cv_scores = np.sqrt(-cross_val_score(linreg, piped_train_set, train_lab, cv=10,
					verbose=True, scoring='neg_mean_squared_error',
					n_jobs=8))
print(
	f'*** CV estimates score of linreg ~{round(np.mean(cv_scores), 2)}'
)

"""
treereg = DecisionTreeRegressor()
treereg.fit(piped_train_set, train_lab)

cv_scores = np.sqrt(-cross_val_score(treereg, piped_train_set, train_lab, cv=10,
					scoring='neg_mean_squared_error', n_jobs=8,
					verbose=True))
print(
	f'*** CV esimates score of treereg ~{round(np.mean(cv_scores), 2)}'
)

forstreg = RandomForestRegressor()
forstreg.fit(piped_train_set, train_lab)

cv_scores = np.sqrt(-cross_val_score(forstreg, piped_train_set, train_lab, cv=10,
					scoring='neg_mean_squared_error', n_jobs=8,
					verbose=True))
print(
	f'*** CV esimates score of forstreg ~{round(np.mean(cv_scores), 2)}'
)
"""

param_dist = {
	'kernel': ['linear', 'rbf'],
	'C': scipy.stats.randint(800, 1000),
	'gamma': scipy.stats.randint(1000, 2000)
}
svr = SVR()
rsearchcv = RandomizedSearchCV(svr, param_dist, scoring='neg_mean_squared_error',
				n_jobs=8, n_iter=3, verbose=True)

rsearchcv.fit(piped_train_set, train_lab)

bestsvr = rsearchcv.best_estimator_.fit(piped_train_set, train_lab)

piped_test_set = pipe.transform(test_set)
pred_test_lab = bestsvr.predict(piped_test_set)
rmse = np.sqrt(mean_squared_error(pred_test_lab, test_lab))
print(
	f'*** Score of SVR against test set ~{round(rmse, 2)}'
)

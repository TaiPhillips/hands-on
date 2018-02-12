import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

oecd_bli = pd.read_csv('./oecd_bli_2015.csv')
"""
Each row of this dataframe represents an indicator with its value for a
specific country. Example:

	>>> oecd_bli.iloc[1231]
	LOCATION                               NOR
	Country                             **Norway**
	INDICATOR                         JE_PEARN
	Indicator                **Personal earnings**
	MEASURE                                  L
	Measure                              Value
	INEQUALITY                             HGH
	Inequality                            High
	Unit Code                              USD
	Unit                             US Dollar
	PowerCode Code                           0
	PowerCode                            Units
	Reference Period Code                  NaN
	Reference Period                       NaN
	Value                                **56046**
	Flag Codes                               E
	Flags                      Estimated value
	Name: 1231, dtype: object

From that you build the _pivot_ table: you scan the table collecting rows
with the same `Country', reading from column `Indicator' and using `Value'
as the value of the pair. That's perfect for this CSV as you have multiple
rows for the same country, each for a different indicator.

Before that you filter rows that have Inequality equals to TOT. Apparently,
the Inequality field of each row tells you whether the row is talking about
a specific category (male, female, high-salary, low-salary, ...) or you're
talking about all of the people. You want all of the people.

What you get in the end is a dataframe where each row is a country, and as
columns it has different indicators - each with its value.

You'll later find out you're interested in `Life satisfaction` only.
"""

oecd_bli = oecd_bli[oecd_bli['INEQUALITY'] == 'TOT']
oecd_bli = oecd_bli.pivot(index='Country', columns='Indicator', values='Value')

gdp_per_capita = pd.read_csv('./gdp_per_capita.csv', delimiter='\t',
				thousands=',', encoding='latin')
gdp_per_capita.set_index('Country', inplace=True)
gdp_per_capita.rename(columns={'2015': 'GDP per capita'}, inplace=True)

oecd_bli = oecd_bli[['Life satisfaction']]
gdp_per_capita = gdp_per_capita[['GDP per capita']]

df = pd.merge(oecd_bli, gdp_per_capita, left_index=True, right_index=True)

# For some reason he decides to remove some indeces.
# Don't know why ...maybe the model will perform better?
df.sort_values(by='GDP per capita', inplace=True)
remove_indeces = [0, 1, 6, 8, 33, 34, 35]
keep_indeces = list(set(range(36)) - set(remove_indeces))
df = df.iloc[keep_indeces]

"""
# You can easily make plots using pd.DataFrame as a front-end
df.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()
"""

# sklearn expects np arrays, see docs
X = df['GDP per capita'].values.reshape((29, 1))
y = df['Life satisfaction'].values

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

fmt = '*** Predicted life satisfaction in Cyprus={:.3f}'
cyprus_gdp = [[22587]]
print(fmt.format(model.predict(cyprus_gdp)[0]))

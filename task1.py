import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv("HealthExpend.csv")

data.isnull().sum()

X = data[["AGE", "famsize", "COUNTIP", "COUNTOP", "EXPENDIP"]]
Y = data["EXPENDOP"]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y,
    test_size = 0.2,
    random_state = 42
)

x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

ols_model = sm.OLS(y_train, x_train).fit()

print(ols_model.summary())

y_pred_train = ols_model.predict(x_train)
y_pred_test = ols_model.predict(x_test)

metrics.r2_score(y_train, y_pred_train)

sns.heatmap(X.corr(), annot=True, vmin = -1, vmax = 1, cmap = "PuOr")

vif_data = pd.DataFrame()
vif_data["features"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


y_real = pd.concat([y_train, y_test], axis = 0)
y_pred = pd.concat([y_pred_train, y_pred_test], axis = 0)
Y_data = {
    "Y_real" : y_real.sort_index(),
    "Y_pred" : y_pred.sort_index()
}
Y_dataframe = pd.DataFrame(Y_data)
Y_dataframe["Error"] = Y_dataframe["Y_real"] - Y_dataframe["Y_pred"]

fig, ax = plt.subplots(figsize=(10,10))

ax.axhline(y = 0, color = "r")
ax.scatter(np.log(y_real), Y_dataframe["Error"])

lambdas = np.arange(0,1.01,0.01)
lambdas

metrics.mean_squared_error(y_pred_test, y_test)

3769074.4565174244

lambdas = np.arange(0,1.01,0.01)
r2_values = []
MSE = []

for l in lambdas:
    ridge_model = sm.OLS(y_train, x_train).fit_regularized(alpha=l, L1_wt=0)
    y_pred_train = ridge_model.predict(x_train)

    r2_values.append(
        metrics.r2_score(y_train, y_pred_train)
    )

    MSE.append(
        metrics.mean_squared_error(y_train, y_pred_train)
    )

coeficients = {
    "Lambda": lambdas,
    "R2_score": r2_values,
    "MSE": MSE
}

coef_frame = pd.DataFrame(coeficients)

print(coef_frame)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

polynomial = PolynomialFeatures(degree=3)

X_train_poly = polynomial.fit_transform(X_train)
X_test_poly = polynomial.transform(X_test)

X_train_poly = sm.add_constant(X_train_poly)
X_test_poly = sm.add_constant(X_test_poly)

nonlinear_model = sm.OLS(Y_train, X_train_poly).fit()

print(ols_model.summary())

X = X.drop(['famsize', 'EXPENDIP'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y,
    test_size = 0.2,
    random_state = 42
)

x_train = sm.add_constant(x_train)
x_test = sm.add_constant(x_test)

ols_model = sm.OLS(y_train, x_train).fit()

y_pred_train = ols_model.predict(x_train)
y_pred_test = ols_model.predict(x_test)

print(ols_model.summary())

import timeit

time_model_odd = timeit.timeit(lambda: ols_model_odd.fit(), number=10000)
time_model_new = timeit.timeit(lambda: ols_model_new.fit(), number=10000)

print("Час для старої моделі:", time_model_odd)
print("Час для нової моделі:", time_model_new)

X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, random_state=42)
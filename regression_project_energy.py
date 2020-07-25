# -*- coding: utf-8 -*-
"""
**Kaggle link**

- Dataset on kaggle - (https://www.kaggle.com/loveall/appliances-energy-prediction)

**Context**

Experimental data used to create regression models of appliances energy use in a low energy building.

**Content**

The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters. Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column. Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters).

**T dew point**

If you are interested in a calculation that gives an approximation of dew point temperature if you know the observed temperature and relative humidity, the following formula was proposed in a 2005 article by Mark G. Lawrence in the Bulletin of the American Meteorological Society:

Td = T - ((100 - RH)/5.)

where Td is dew point temperature (in degrees Celsius), T is observed temperature (in degrees Celsius), and RH is relative humidity (in percent). Apparently this relationship is fairly accurate for relative humidity values above 50%.

# Import and Load data
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

from datetime import datetime
from enum import Enum
from itertools import combinations, product

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

FIGURE_SIZE = (15, 12)
SNS_FIGURE_SIZE = (20, 15)
RANDOM_STATE = 42
TEST_SIZE = 0.2

file_path = 'KAG_energydata_complete.csv'
loaded_df = pd.read_csv(file_path, parse_dates=['date'])

features_df = loaded_df.drop(['rv1', 'rv2','lights', 'Appliances'], axis = 1)
target_column = loaded_df['Appliances'] + loaded_df['lights'] # Both represent energy use in Wh

"""# View data

First lets take a look at the shape of our dataframe, to see how many potential features we have and the size of our infomation (observations)
"""

features_df.shape

"""So we have 19735 row, meaning that we have enough information for our regression

But we have 25 potential features! that could easily cause our model to be overfitted - we'll have to reduce this number

### Missing values

Lets check for missing values in our dataframe
"""

loaded_df.isnull().sum()

"""No missing values!

### Correlation map

Next we'll check for correlation between our features, as our current goal is reduce the number of features
"""

fig, ax = plt.subplots(figsize=SNS_FIGURE_SIZE)
mask_matrix = np.triu(features_df.corr())
sns.heatmap(features_df.corr(), annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', mask=mask_matrix)

"""If we set the "correlation limit" to 0.8, we get 4 potential groups of variables with high correlation and therefore redundent"""

correlated_column_group_1 = ['T1', 'T2', 'T3', 'T4', 'T5', 'T7', 'T8', 'T9']
correlated_column_group_2 = ['RH_1', 'RH_2', 'RH_3', 'RH_4', 'RH_7']
correlated_column_group_3 = ['T6', 'T_out'] # Tdewpoint could be added (correlation of 0.79)
correlated_column_group_4 = ['RH_8', 'RH_9']

"""### More viewing options"""

features_df.info()

features_df.describe()

features_df.head()

features_df.tail()

features_df.sample(5)

"""# Feature enginnering

## Extracting time features

As we go about our exploration, we'll collect irrelevant features, meaning features that doesn't increase our information.

we'll add the column 'Date' since it can't be used in the final regression
"""

irrelevant_features = []
irrelevant_features.append('date')

"""In order to study our features further, we'll increase our resolution by expending the 'date' column into 3 columns:
1. Day of week - could serve as a categorical feature if needed, since weekdays and weekends are likely to yield different energy consumption.
2. Month - could serve as a categorical feature if needed, since different months of the year are also likely to yield different energy consumption.
3. Hour - will be our lowest resolution for the exploration.

This feature extraction will be done using a custom function, that takes a dataframe with a column labeled 'date' and adds the needed feature we've mentioned

### time features extractor
"""

def time_features_extractor (df, attr_list = ('year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek')):
    for attr in attr_list:
        df[attr] = df['date'].apply(lambda x: getattr(x, attr))
    return df

features_df = time_features_extractor(features_df, ('dayofweek', 'month', 'hour'))
features_df.sample(5)

"""In order to transform the column we've created 'dayofweek' to a binary column between weekdays and weekends, we'll divide by 5 without resdiual, and add 'dayofweek' to our irrelavent features list"""

features_df['weekday'] = features_df['dayofweek'].apply(lambda x: x//5)
irrelevant_features.append('dayofweek')

"""### Creating group by objects

Next, to ease up the rest of the exploration, we'll create two group by objects, one for month and one for day
"""

gb_month = features_df.groupby(['month'])
gb_hour = features_df.groupby(['hour'])

"""## Confirming and dealing with correlated features

### correlated column group 1
"""

ccgroup1_gb_month = gb_month[correlated_column_group_1]
ccgroup1_gb_hour = gb_hour[correlated_column_group_1]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=FIGURE_SIZE)
ccgroup1_gb_month.mean().plot(ax=ax1)
ccgroup1_gb_hour.mean().plot(ax=ax2)

ax1.set_ylabel('Temprature[C]')
ax2.set_ylabel('Temprature[C]')

"""From this view, we can see the besides T2, most of those features shows a very similar trend and a small variety of values (around 1 degree), but T2 shows a different trend from the rest, and offers a wider variety of values(up to 3 degrees)

we'll first create a function to average the correlated features and then we'll add the irrelevant results to our irrelevant features list
"""

def features_average (df, features_to_average, new_feature_name):
    df[new_feature_name] = np.mean(df[features_to_average], axis=1)
    return df

correlated_column_group_1.remove('T2')
irrelevant_features = irrelevant_features + correlated_column_group_1
features_df = features_average(features_df,
                               features_to_average = correlated_column_group_1,
                               new_feature_name = 'T_avg_1_3_4_5_7_8_9')
features_df.sample(5)

"""### correlated column group 2"""

ccgroup2_gb_month = gb_month[correlated_column_group_2]
ccgroup2_gb_hour = gb_hour[correlated_column_group_2]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=FIGURE_SIZE)
ccgroup2_gb_month.mean().plot(ax=ax1)
ccgroup2_gb_hour.mean().plot(ax=ax2)

ax1.set_ylabel('Humidity[%]')
ax2.set_ylabel('Humidity[%]')

"""Again, we can see that we have a group of features that acts similarly and one "outlair" ('RH_7')

We'll deal with it in the same manner
"""

correlated_column_group_2.remove('RH_7')
irrelevant_features = irrelevant_features + correlated_column_group_2
features_df = features_average(features_df,
                               features_to_average = correlated_column_group_2,
                               new_feature_name = 'RH_avg_1_2_3_4')
features_df.sample(5)

"""### correlated column group 3"""

ccgroup3_gb_month = gb_month[correlated_column_group_3]
ccgroup3_gb_hour = gb_hour[correlated_column_group_3]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=FIGURE_SIZE)
ccgroup3_gb_month.mean().plot(ax=ax1)
ccgroup3_gb_hour.mean().plot(ax=ax2)

ax1.set_ylabel('Temprature[C]')
ax2.set_ylabel('Temprature[C]')

"""These two features are almost identical - so we'll unite them"""

irrelevant_features = irrelevant_features + correlated_column_group_3
features_df = features_average(features_df,
                               features_to_average = correlated_column_group_3,
                               new_feature_name = 'T_avg_6_out')
features_df.sample(5)

"""### correlated column group 4"""

ccgroup4_gb_month = gb_month[correlated_column_group_4]
ccgroup4_gb_hour = gb_hour[correlated_column_group_4]

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=FIGURE_SIZE)
ccgroup4_gb_month.mean().plot(ax=ax1)
ccgroup4_gb_hour.mean().plot(ax=ax2)

ax1.set_ylabel('Humidity[%]')
ax2.set_ylabel('Humidity[%]')

"""These two features are almost identical - so we'll unite them"""

irrelevant_features = irrelevant_features + correlated_column_group_4
features_df = features_average(features_df,
                               features_to_average = correlated_column_group_4,
                               new_feature_name = 'RH_avg_8_9')
features_df.sample(5)

"""### Summary and final exclusion

Now that we've found our irrelevant features, we'll exclude them from df and check again how many features we have now
"""

irrelevant_features.append('hour')

chosen_features_df = features_df.drop(irrelevant_features, axis=1)
chosen_features_df.shape

"""So we're down to 15 features! much better! 

Lets take another look at the correlation map to see whats the current situation
"""

fig, ax = plt.subplots(figsize=SNS_FIGURE_SIZE)
mask_matrix = np.triu(chosen_features_df.corr())
sns.heatmap(chosen_features_df.corr(), annot = True, cmap= 'coolwarm', linewidths=3, linecolor='black', mask=mask_matrix)

"""## Preprocessing

Before we'll fit our selected features to the model, we'll preprocess(mostly normalization) our data.

Lets first divide the features into categories, so we can select different preprocessing for each group
"""

temperature_columns = [column_name for column_name in chosen_features_df.columns if column_name.startswith('T')]
humidity_columns = [column_name for column_name in chosen_features_df.columns if column_name.startswith('RH')]
environment_columns = ['Windspeed', 'Press_mm_hg', 'Visibility']

"""For the temprature and humidity columns, we'll use a global min/max scaler. meaning that we'll take the global min/max from all temperature/humidity columns, and use those limits for our scaling the data between 0 to 1

To that purpose we'll bulit a custom transformer.
"""

class CustomMinMaxScaler (BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale

    def fit(self, X, y=None):
        self.min_value_ = self._get_global_min(X[self.columns_to_scale])
        self.max_value_ = self._get_global_max(X[self.columns_to_scale])
        return self

    def transform(self, X, *_):
        norm_value = self.max_value_ - self.min_value_ 
        X[self.columns_to_scale] = X[self.columns_to_scale].apply(lambda x: (x-self.min_value_)/norm_value)
        return X

    def _get_global_max(self, part_X):
        return max(np.max(part_X))

    def _get_global_min(self, part_X):
        return min(np.min(part_X))

"""Lets test them out"""

scaled_features_df = chosen_features_df.copy()

temperature_scaler = CustomMinMaxScaler(columns_to_scale = temperature_columns)
temperature_scaler.fit_transform(scaled_features_df)

humidity_scaler = CustomMinMaxScaler(columns_to_scale = humidity_columns)
humidity_scaler.fit_transform(scaled_features_df)

"""Great! for the rest of the columns, we'll scale them in the same range of our other columns, so we'll use the standard MinMaxScaler"""

min_max_scaler = MinMaxScaler()
scaled_features_df['Press_mm_hg'] = min_max_scaler.fit_transform(np.array(chosen_features_df['Press_mm_hg']).reshape(-1,1))
scaled_features_df['Windspeed'] = min_max_scaler.fit_transform(np.array(chosen_features_df['Windspeed']).reshape(-1,1))
scaled_features_df['Visibility'] = min_max_scaler.fit_transform(np.array(chosen_features_df['Visibility']).reshape(-1,1))

"""Good, so now we have our features chosen and scaled, we'll set the final groups and create a column transformer to orginize the results neatly."""

class ColumnsDevision(Enum):
    final_temperature_columns = [column_name for column_name in scaled_features_df.columns if column_name.startswith('T')]
    final_humidity_columns = [column_name for column_name in scaled_features_df.columns if column_name.startswith('RH')]
    final_environment_columns = ['Press_mm_hg', 'Windspeed', 'Visibility']

column_transformer = make_column_transformer(
                        (CustomMinMaxScaler(columns_to_scale = ColumnsDevision.final_temperature_columns.value), ColumnsDevision.final_temperature_columns.value),
                        (CustomMinMaxScaler(columns_to_scale = ColumnsDevision.final_temperature_columns.value), ColumnsDevision.final_humidity_columns.value),
                        (MinMaxScaler(), ColumnsDevision.final_environment_columns.value),
                        )

"""# Model selection and optimization

## Inital model selection

First, we'll split the data into train/test sets, by ratio of 4:1 accordingly, or 80%-20% split
"""

regression_df = scaled_features_df.copy()
X_train, X_test, y_train, y_test = train_test_split(regression_df, target_column, test_size=TEST_SIZE, random_state=RANDOM_STATE) # stratify=y

"""For initial test, we'll take the 3 models that we've learned so far and run them on our dataset, in order to choose on which model to foucs our hyper-parameter tuning"""

models = [
           ['Linear regression: ', LinearRegression()],
           ['Desicion tree regressor: ', DecisionTreeRegressor()],
           ['KNeighbors regressor: ',  KNeighborsRegressor()],
         ]

rmse_loss = lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))
model_data = []
for name,curr_model in models :
    curr_model_data = {}
    curr_model_data["Name"] = name
    curr_model.fit(X_train, y_train)
    curr_model_data["Train_RMSE_Score"] = round(rmse_loss(y_train, curr_model.predict(X_train)),2)
    curr_model_data["Train_R^2_Score"] = round(r2_score(y_train, curr_model.predict(X_train)),2)
    curr_model_data["Test_RMSE_Score"] = round(rmse_loss(y_test,curr_model.predict(X_test)),2)
    curr_model_data["Test_R^2_Score"] = round(r2_score(y_test,curr_model.predict(X_test)),2)
    model_data.append(curr_model_data)

model_data

"""Lets analyze the results:
1. Linear regression - very poor results on both the train and test sets, makes sense since we have a few categorical veriables which tends to hinder this regression.
2. Desicion tree regressor = seems to casue over fitting on the train set(R^2=1), and a poor score on the test set as a result.
3. K nearest neighbors regressor - medium results on both the train/test sets, but since it's the best out of the three we'll focus on it.

We have our regressor, but the results are still far from being optimized, which means we should go back to feature enginnering in order to improve our results

In order to asses the value of RMSE, we can look at the mean value of the target column and it's distribution
"""

target_column.hist()

target_column.mean()

"""## Feature selection

Now that we've chosen a model, we'll need to optimize our feature selection.
We have 15 features - the number of possible combinations for feature selection is around 30k.. so in order to choose the right combination, we'll construct a series of functions designed to get all possible combinations (using itertools) and evaluate each one of them in our model.

The basic algorithim is as follows:
1. select a number of features (from 15 to 5)
2. find all possible combinations for that number of features
3. for each possible combination, run KNN regression
4. calculate the RMSE and R^2 for both train and test datasets
5. pick a different number of features and repeat steps 2-4 till all numbers between 5-15 were selected

We'll save the log and details in a dictionary call "models_data", and after we'll attempt to find the best 10 feature selections, by finding the minimum RMSE and the maximum R^2

After we'll have all that information, we can select a feature list for further optimization

**Note:** since we're doing a step-by-step optimization, we are at risk of missing the global optimal point and landing in local optimal point instead
"""

class FeatureSelector ():
    def get_model_data (self, reg_model, X_train, X_test, y_train, y_test):
        model_data = {}
        model_data['features'] = set(X_train.columns)
        model_data['parameters'] = reg_model.get_params()
        reg_model.fit(X_train, y_train)
        y_hat_train = reg_model.predict(X_train)
        y_hat_test = reg_model.predict(X_test)
        model_data['Train_RMSE_score'] = round(rmse_loss(y_train, y_hat_train),2)
        model_data['Train_R^2_score'] = round(r2_score(y_train, y_hat_train),2)
        model_data['Test_RMSE_score'] = round(rmse_loss(y_test, y_hat_test),2)
        model_data['Test_R^2_score'] = round(r2_score(y_test, y_hat_test),2)
        return model_data

    def build_and_test_model (self, reg_model, df, target_column):
        X_train, X_test, y_train, y_test = train_test_split(df, target_column, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        return self.get_model_data (reg_model, X_train, X_test, y_train, y_test)

    def feature_iterator (self, df, target_column, reg_model, min_features_num, max_features_num):
        models_data = {}
        for features_num in range(min_features_num, max_features_num):
            possible_features = set(combinations(df.columns, features_num))
            for inner_index, selected_features in enumerate(possible_features):
                model_name = 'reg_model_' + str(features_num) + '_' + str(inner_index)
                models_data[model_name] = self.build_and_test_model (reg_model, df[list(selected_features)], target_column)
        return models_data

    def get_n_best_models(self, models_data, key_parameter, result_goal, n_results):
        best_models_names = []
        temp_models_data = models_data.copy()
        for index in range(n_results):
            if result_goal == 'max':
                current_best_model_name = self.get_model_max_value(temp_models_data, key_parameter)
            elif result_goal == 'min':
                current_best_model_name = self.get_model_min_value(temp_models_data, key_parameter)
            best_models_names.append(current_best_model_name)
            del temp_models_data[current_best_model_name]
        return best_models_names

    def get_model_max_value (self, models_data, key_parameter):
        max_value = 0
        for name, model in models_data.items():
            if model[key_parameter] > max_value:
                max_value = model[key_parameter]
                model_name = name
        return model_name

    def get_model_min_value (self, models_data, key_parameter):
        min_value = 1000000
        for name, model in models_data.items():
            if model[key_parameter] < min_value:
                min_value = model[key_parameter]
                model_name = name
        return model_name

    def models_data_to_df (self, models_data, models_list, selected_keys):
        temp_list = []    
        for model_name in models_list:
            temp_dict = {key: value for key, value in models_data[model_name].items() if key in selected_keys}
            temp_list.append(temp_dict)
        result_df = pd.DataFrame(temp_list, index=models_list)
        return result_df

    def set_parameters (self, reg_model, parameters_name, parameters_value):
        for index, parameter_name in enumerate(parameters_name):
            setattr(reg_model, parameter_name, parameters_value[index])
        return reg_model
    
    def tune_models_parameters (self, df, target_column, reg_model, models_data, models_list, parameters_to_tune):
        parameter_matrix = product(*parameters_to_tune.values())
        new_models_data = models_data.copy()
        for model_name in models_list:
            selected_features = models_data[model_name].get('features')
            for inner_index, parameters in enumerate(parameter_matrix):
                reg_model = self.set_parameters(reg_model, parameters_to_tune.keys() ,parameters)
                new_model_name = model_name + '_' + str(inner_index)
                new_models_data[new_model_name] = self.build_and_test_model (reg_model, df[list(selected_features)], target_column)
        return new_models_data

feature_selector = FeatureSelector()

# number of combination - 15 choose 5 + 15 choose 6 + 15 choose 7 + ..
models_data = feature_selector.feature_iterator(scaled_features_df, target_column, KNeighborsRegressor(), 5, len(scaled_features_df.columns))

models_data

"""Now that we have the results, we can find the optimal combinations for RMSE/R^2"""

selected_keys = ('Test_R^2_score', 'Test_RMSE_score', 'Train_R^2_score', 'Train_RMSE_score')
selected_models = feature_selector.get_n_best_models(models_data, 'Test_R^2_score', 'max', 10)
result_df = feature_selector.models_data_to_df(models_data, selected_models, selected_keys)
result_df

features_opt1_df = scaled_features_df[models_data[selected_models[0]].get('features')]

selected_keys = ('Test_R^2_score', 'Test_RMSE_score', 'Train_R^2_score', 'Train_RMSE_score')
selected_models = feature_selector.get_n_best_models(models_data, 'Test_RMSE_score', 'min', 10)
result_df = feature_selector.models_data_to_df(models_data, selected_models, selected_keys)
result_df

features_opt2_df = scaled_features_df[models_data[selected_models[0]].get('features')]

"""## Hyperparameter finetuning using GridSearchCV

### Setting up and running the search

Our first attempt to finetune the models parameter will be using GridSearchCV.
In order to do so, we'll need to carefully define parameter ranges and scoring methods
"""

r2_scorer = make_scorer (r2_score)
rmse_scorer = make_scorer (rmse_loss)

leaf_size_range = [10, 20, 30, 40, 50, 60]
n_neighbors_range = [3, 5, 7, 9, 11]
p_range = [1, 2]
        
tuned_params = {
    'leaf_size': leaf_size_range,
    'n_neighbors': n_neighbors_range,
    'p' : p_range
               }

scoring = {'RMSE' : rmse_scorer,
           'R^2' : r2_scorer}

"""Now that everything is set, we can define the grid search object and use it to optimize our model"""

gs = GridSearchCV(estimator = KNeighborsRegressor(), param_grid = tuned_params, scoring= scoring, refit='RMSE', cv=5, return_train_score=True)

gs.fit(features_opt1_df, target_column)
features_opt1_best_params = gs.best_params_

gs.fit(features_opt2_df, target_column)
features_opt2_best_params = gs.best_params_

"""### the results and final selection"""

features_opt1_best_params

features_opt2_best_params

cv_results_df = pd.concat([pd.DataFrame(gs.cv_results_['params']),
                           pd.DataFrame(gs.cv_results_['mean_test_RMSE'], columns=['RMSE']),
                           pd.DataFrame(gs.cv_results_['mean_test_R^2'], columns=['R^2'])] ,axis=1)
cv_results_df = cv_results_df.sort_values(['RMSE'], axis=0, ascending=False)
cv_results_df.head(10)

"""To see the actual results of each model, we'll now run each model-df couple with its parameters"""

opt_clf_1 = KNeighborsRegressor()
for parameter, value in features_opt1_best_params.items():
    setattr(opt_clf_1, parameter, value)
feature_selector.build_and_test_model(opt_clf_1, features_opt1_df, target_column)

opt_clf_2 = KNeighborsRegressor()
for parameter, value in features_opt2_best_params.items():
    setattr(opt_clf_2, parameter, value)
feature_selector.build_and_test_model(opt_clf_2, features_opt2_df, target_column)

"""## Hyperparameter finetuning using FeatureSelector (custom)

Alternatively, we can use the FeatureSelector we've built here

In order to expend out search and lower the chance that we've landed on a local optimum point, we'll now search the "top" models with different features, and for each of these models tune the parameters for optimum results

To that purpose, we'll use our FeatureSelector

### Setting up and running the search
"""

selected_models = feature_selector.get_n_best_models(models_data, 'Test_RMSE_score', 'min', 10) + feature_selector.get_n_best_models(models_data, 'Test_R^2_score', 'max', 10)

updated_models_data = feature_selector.tune_models_parameters (scaled_features_df, target_column, KNeighborsRegressor(), models_data, selected_models, tuned_params)

"""### the results and final selection"""

selected_keys = ('Test_R^2_score', 'Test_RMSE_score', 'Train_R^2_score', 'Train_RMSE_score')
selected_models = feature_selector.get_n_best_models(updated_models_data, 'Test_RMSE_score', 'min', 10) + feature_selector.get_n_best_models(updated_models_data, 'Test_R^2_score', 'max', 10)
result_df = feature_selector.models_data_to_df(updated_models_data, selected_models, selected_keys)
result_df

features_opt3_df = scaled_features_df[updated_models_data[selected_models[0]].get('features')]
features_opt3_best_params = updated_models_data[selected_models[0]].get('parameters')
opt_clf_3 = KNeighborsRegressor()
for parameter, value in features_opt3_best_params.items():
    setattr(opt_clf_3, parameter, value)
feature_selector.build_and_test_model(opt_clf_3, features_opt2_df, target_column)

"""# Final regressor (for production level)"""

class ClassiferContsants(Enum):   
    N_NEIGHBORS = 3
    LEAF_SIZE = 10
    P = 2

class KnnRegressor (BaseEstimator, ClassifierMixin):  
    def __init__(self,
                 n_neighbors = ClassiferContsants.N_NEIGHBORS.value,
                 leaf_size = ClassiferContsants.LEAF_SIZE.value,
                 p = ClassiferContsants.P.value,
                ):
      self.n_neighbors = n_neighbors
      self.leaf_size = leaf_size
      self.p = p

    def _pipeline_constructor(self):
        self.pipeline_ = Pipeline(steps=[
                                       ('classifier', KNeighborsRegressor(
                                                                          n_neighbors = self.n_neighbors,
                                                                          leaf_size = self.leaf_size,
                                                                          p = self.p,
                                                                                                 ),
                                       )])

    def fit(self, X, y=None):  
        self._pipeline_constructor()
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.pipeline_.predict(X)

    def score(self, X, y=None):
        rmse_score = lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))
        rmse, r2 = round(rmse_score(X, y), 2), round(r2_score(X, y), 2)
        print(f"RMSE = {rmse}\t R^2 = {r2}")
        return rmse, r2

clf = KnnRegressor()
print('Model 1 results:')
regression_df = features_opt1_df.copy()
X_train, X_test, y_train, y_test = train_test_split(regression_df, target_column, test_size=TEST_SIZE, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)
print ('Train results')
clf.score(y_train, clf.predict(X_train))
print ('Test_results')
clf.score(y_test, clf.predict(X_test))

print('\nModel 2 results:')
regression_df = features_opt2_df.copy()
X_train, X_test, y_train, y_test = train_test_split(regression_df, target_column, test_size=TEST_SIZE, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)
print ('Train results')
clf.score(y_train, clf.predict(X_train))
print ('Test_results')
clf.score(y_test, clf.predict(X_test))

print('\nModel 3 results:')
regression_df = features_opt3_df.copy()
X_train, X_test, y_train, y_test = train_test_split(regression_df, target_column, test_size=TEST_SIZE, random_state=RANDOM_STATE)
clf.fit(X_train, y_train)
print ('Train results')
clf.score(y_train, clf.predict(X_train))
print ('Test_results')
clf.score(y_test, clf.predict(X_test))

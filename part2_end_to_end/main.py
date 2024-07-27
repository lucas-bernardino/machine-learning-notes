import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv("housing.csv")

def get_general_information(): 
    desc = df.describe()
    print("\n\t\t\t\t\t\t\tGENERAL INFORMATION ABOUT THE DATASET\n\n", desc)
    
    df.hist(bins=50, figsize=(20, 15))
    plt.show()

# We need to split our dataset in a training and testing set. Usually, 20% of the data is test and the other is training. We must do that
# randomly, that's why we call this function from sklearn.
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

def get_biased_strat_sampling(): 
    # Suppose that median income is a very important attribute to predict median house pricing. So, we need to ensure that
    # the test set is representative of the various categories of incomes in the whole dataset. 
    # We can create an income category with 5 categories, according to the median income.
    df["income_cat"] = pd.cut(df["median_income"], bins=[0, 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])

    # Now, we can do a stratified sampling based on the income category by using the StratifiedShuffleSplit class from sklearn.
    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    strat_train_set = strat_test_set = df.loc[0]
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    # Remove the income_cat attribute from the DataFrame
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set
    
def visualizing_geographical_data():
    df_train_copy, _ = get_biased_strat_sampling()
    
    # We can look at house pricings with the following code.
    # The radius of each circle represents the district's population (option s), and the color represents the price (option c)
    # The color map is obtained from the option cmap called jet, which ranges from blue - low values - to red - high prices.
    df_train_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=df_train_copy["population"]/100, 
                       label="population", c=df_train_copy["median_house_value"], cmap=plt.get_cmap("jet"), colorbar=True)
    plt.legend()
    plt.show()

def getting_correlations():
    df_train, _ = get_biased_strat_sampling()
    df_train_copy = df_train.drop(columns='ocean_proximity') # Since it's made up of words and not numbers
    
    # We can look at hou much each attribute correleates with the median house value. It ranges from -1 to 1. 
    # When it's close to 1, it means that there is a strong positive correlation.
    # When it's close to -1, it means there is a strong negative correlation.
    # When it's close to zero, it means that there's no linear correlation.
    corr_matrix = df_train_copy.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # We can also use scatter_matrix function from pandas to visualize every numerical attribute against
    # every other numerical attribute.
    
    # from pandas.plotting import scatter_matrix
    # df_columns_list = df_train_copy.columns.values
    # scatter_matrix(df_train_copy[df_columns_list])
    # plt.show()

    # The most promising attribute to predict the median house value is the median income. Let's plot their correlation
    # We can see that there's a horizontal line around $450,000 and another around $350,000.
    # Removing the corresponding districts would prevent the algorithm from learning to reproduce these data quirks.
    df_train_copy.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.show()

    
    # We can also play around with our dataset and get more mathematical information.
    # For instance, the total number of rooms in a distrct is novery useful if we don't know how many households are there.
    # What we really want is the number of rooms per household. Similarly, the total number of bedrooms by itself is not 
    # very useful: we probably want to compare it to the number of rooms. And the population per household also semms
    # like an interesting attribute combination to look at. We can create these attributes:
    df_train_copy["rooms_per_household"] = df_train_copy["total_rooms"] / df_train_copy["households"]
    df_train_copy["bedrooms_per_room"] = df_train_copy["total_bedrooms"] / df_train_copy["total_rooms"]
    df_train_copy["population_per_household"] = df_train_copy["population"] / df_train_copy["households"]

    corr_matrix = df_train_copy.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # We can see that our new attributes are much more correlated with the median house than the ones we had before.

def clean_data():
    df_train, _ = get_biased_strat_sampling()
    housing = df_train.drop("median_house_value", axis=1)
    housing_labels = df_train["median_house_value"].copy()

    # Most ML algorithms can't work with missing features. The column total_bedrooms has some missing values. There are 
    # a few ways we can fix this:
    # * Get rid of the corresponding districts
    # * Get rid of the whole attribute
    # * Set the values to some value (zero, the mean, the median, etc)
    
    # housing.dropna(subset=["total_bedrooms"])  ## option 1
    # housing.drop("total_bedrooms", axis=1)  ## option 2
    # median = housing["total_bedrooms"].median()  ## option 3
    # housing["total_bedrooms"].fillna(median, inplace=True)  ## option 3

    
    # There's a built-in class that takes care of missing values in sklearn: SimpleImputer
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    
    # We can only use numerical attributes, so we need to make a copy by excluding the ocean_proximity, which is made up of words.
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)

    # We can use this trained imputer to transform the trainong set by replacing missing  values by the learned medians:
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    
    
    # Remember that we left out the column ocean_proximity because it's a text attribute. Let's convert it from text to numbers 
    # by using Scikit-Learn's OrdinalEncoder class:
    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    housing_text_column = housing[["ocean_proximity"]]
    housing_text_converted = ordinal_encoder.fit_transform(housing_text_column)

    
    # There's just an issue with this representation, where categories 0 and 4 are more similar than 0 and 1.
    # To fix it, we can create a binary attribute per category, where one attribute equal to 1 when the category 
    # is <IH OCEAN (and 0 otherwise), another attribute equal to 1 when the category is INLAND (and 0 otherwise) 
    # and so on. This is called one-hot encoding:
    from sklearn.preprocessing import OneHotEncoder
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_text_column)




    # That was an example of a Transformer. However, often we'll need to write our own transformer in order to do a custom cleanup.
    # And for that, we need our transformer to work seamlessly with Scikit-Learn functionalities. To do that, we basically need
    # to create a class and implement three methods: fit() (returning self), transform(), and fit_transform(). We get the last 
    # one for free by simply adding TransformerMixin as a base class. Similarly, if we add BaseEstimator as a base class, we get 
    # two extra methods (get_params() and set_params()) that can be useful later on.
    # This is an example of what a small transformer class that adds some combined attributes we've talked before:

    from sklearn.base import BaseEstimator, TransformerMixin
    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin): 
        def __init__(self, add_bedrooms_per_room = True):
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[: , bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else: 
                return np.c_[X, rooms_per_household, population_per_household]

    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
    housing_extra_attrbs = attr_adder.transform(housing.values)


    return housing_num


def feature_scaling():
    # Machine Learning Algorithms don't perform well when the input numerical attributes have very different scales. 
    # This is the case for the housing data: the total number of rooms ranges from about 6 to 39 320, while 
    # the median income only range from 0 to 15.
    
    # There are two common ways to get all attributes to have the same scale: min-max scaling (normalization) and standardization.
    
    # Min-max scaling (also called as normalization) works like this: valures are shifted and rescaled so that 
    # they end up ranging from 0 to 1. This is done by subtracting the min value and diving by the max minus the min.
    # There's a built-in transformer in Scikit-Learn called MinMaxScaler.
    
    # Standardization is different. It works by first subtracting the mean value (so standardized values always 
    # have a zero mean), and then it divides by the standard deviation so that the resulting distribution has unit variance.
    # Unlike min-max, standardization does not bound values to a specific range, which can be a problem for some algs. like NN,
    # which expect values from 0 to 1. The benefit of standardization is that it's less affect by outliers.
    print("No code for this part, only important theory :)")

def transformation_pipelines():
    # Often, transformations need to be executed in the right order. There's a Scikit-Learn class called Pipeline to help
    # with these sequences:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.base import BaseEstimator, TransformerMixin

    rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
    housing_num = clean_data()

    class CombinedAttributesAdder(BaseEstimator, TransformerMixin): 
        def __init__(self, add_bedrooms_per_room = True):
            self.add_bedrooms_per_room = add_bedrooms_per_room
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
            population_per_household = X[:, population_ix] / X[:, households_ix]
            if self.add_bedrooms_per_room:
                bedrooms_per_room = X[: , bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
            else: 
                return np.c_[X, rooms_per_household, population_per_household]

    

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

    housing_num_tr = num_pipeline.fit_transform(housing_num)


    # So far, we have handled the text columns and the numerical columns separetely. It'd be better if we 
    # could handle all columns, applying the appropriate transformations to each column.
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    df_train, _ = get_biased_strat_sampling()
    housing = df_train.drop("median_house_value", axis=1)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])
    housing_prepared = full_pipeline.fit_transform(housing)

    # That's it. We got the data, explored it, sampled a training set and a test set and we've 
    # written the transformation pipelines to clean up and prepare our data for ML Algorithms.
    # We're now ready to select and train a Machine Learning model.
    return housing_prepared, full_pipeline


def training_model():
    from sklearn.linear_model import LinearRegression
    housing_prepared, _ = transformation_pipelines()

    strat_train_set, _ = get_biased_strat_sampling()
    housing_labels = strat_train_set["median_house_value"].copy()
    
    # The .fit function fits the linear model and usually takes two parameters: X_train and Y_train.
    # In our case, the X_trains is a matrix of shape (16512, 16), meaning that it contains 16512 data points and each point has 16 features.
    # The Y_train is a matrix of shape (16512,), which represents the target data. Note that the length of Y must match the number of rows in X. 
    # Also, in this case we have a multiple linear regression since we have multiple features instead of just one. So, instead of finding a 
    # f(x) = ax + b, we would have f(x1, x2,..., x15) = a1*x1 + b1 + a2*x2 + b2 ...
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    

    # We can measure how accurate it is
    from sklearn.metrics import mean_squared_error
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)    
    # Prints 68627.87, meaning that the typical prediction error is about $68 628, which is not very good since most districts prices
    # range between $120 000 and $265 000.

    # Since the error is high, this can mean that the features do not provide enough information to make good predictions or that the model 
    # is not good enoiugh. We could try to add more features, but let's first try a more complex model:
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    # LEt's evaluate it on the training set:
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    # Prints 0.0. This doesn't mean that the model is perfect; it's much more likely that the model has badly overfit the data. One way to 
    # check this is to use part of the training set for training, and part for model validation.

    

    # We can better evaluate using cross-validation. One way to evaluate the Decision Tree would be to use the train_test_split function 
    # to split the training set into a smaller training set and a validation set, then traing the model against the smaller training set 
    # and evaluate them against the validation set. There's a built-in function that does this under the hood.
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    # tree_rmse_scores: 
    # [73114.78069677 70314.49830377 68080.22869019 71851.95389678 70735.00698724 77456.2757269 71949.14981162 72764.17481871 67133.63212643 72061.85288086]
    # tree_rmse_scores mean: 71581 and tree_rmse_scores standard deviation: 2690

    # That means that the Decision Tree has a score of about 71 581, generally +- 2 690. 
    # Lets compute the same scores for the Linear Regression:
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    # lin_rmse_scores mean: 69104 and lin_rmse_scores standard deviation: 2880
    # The Decision Tree model is overfitting so badly that it performs worse than the Linear Regression model.
    

    # Let's try the last model: RandomForestRegressor. It works by training many Decision Trees on random subsets of the features, then 
    # averaging out their predictions.
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    forest_reg_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    forest_reg_rmse_scores = np.sqrt(-forest_reg_scores) 
    # forest_rmse: 18741.13
    # forest_rmse_scores mean: 50317.07 and forest_rmse_scores standard deviation: 2219.52
    # These results are much better. However, notice that the score on the training set is still much lower than on the validation sets, 
    # meaning that the model is still overfitting the training set.
    

    return lin_reg, tree_reg, forest_reg


def fine_tuning():
    housing_prepared, _ = transformation_pipelines()

    strat_train_set, _ = get_biased_strat_sampling()
    housing_labels = strat_train_set["median_house_value"].copy()

    # Let's assume we have some models and now we need to fine-tune them. We can do that in a few ways.
    # Grid Search: One way to do that would be to play with the hyperparmaeters manually until we find a good combination of them.
    # Instead, we can use GridSearchCV to do that for us. We only need to tekk which hyperparmaeters we want to experiment and 
    # what values tro try out and it will automatically evaluate all the possible combinations of them, using cross-validation.
    # PS: These hyperparameters are not learned from the data but instead are specified by the practitioner to optimize the performance of the model.

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    
    params_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
    ]

    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg, params_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=4)
    grid_search.fit(housing_prepared, housing_labels)

    # Tip: When we have no idea what value a hyperparmeter should have, a simple approach is to try out consecutive powers of 10 or 
    # a smaller number if you want a more fine-grained search.
    # The param_grid will make scikit-learn first evaluate 3 x 4 = 12 combinations of n_estimators and max_features and then 
    # try all 2 x 3 = 6 combinations of hyperparameters in the second dict.
    # That means that the grid search will explore 12 + 6 = 18 combinations of hyperparameters for the RandomForestRegressor, 
    # training each model five times. So there will be 18 x 5 = 90 rounds of training.

    best_params = grid_search.best_params_ # Best combination, which is {'max_features': 8, 'n_estimators': 30}
    best_estimator = grid_search.best_estimator_ # Best estimator, which outputs the whole class of RandomForestRegressor with the best hyperparams.


    # The drid search approach is fine when we're exploring relatively few combinations. However, when the hyperparameter search page 
    # is large, it's often preferable to use RandomizedSearchCV. It's similar to GridSearchCv, but instead of trying out all possible 
    # combinations, it evaluates a given number of random combinations by selecting a random value for each hyperparameter at every iteration.


    return best_estimator

def evaluate():

    # Now, it's time to evaluate the final model on the test set. The process is simple: get the predictors and the labels from our test set,
    # run our full_pipeline to transform the data (call transform(), not fit_transform(), we don't want to fit the test set), and evaluate the final model
    # on the test set.

    final_model = fine_tuning()
    _, strat_test_set = get_biased_strat_sampling()
    
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    _, full_pipeline = transformation_pipelines()
    X_test_prepared = full_pipeline.transform(X_test)

    final_predictions = final_model.predict(X_test_prepared)

    from sklearn.metrics import mean_squared_error

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse) # 48135.26



if __name__ == "__main__":
    evaluate()














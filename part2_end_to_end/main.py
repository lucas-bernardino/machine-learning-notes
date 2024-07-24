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
    return housing_prepared


def training_model():
    from sklearn.linear_model import LinearRegression
    housing_prepared = transformation_pipelines()

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




if __name__ == "__main__":
    training_model()














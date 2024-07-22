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


if __name__ == "__main__":
    getting_correlations()

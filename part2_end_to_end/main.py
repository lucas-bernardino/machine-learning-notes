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
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    # Remove the income_cat attribute from the DataFrame
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)


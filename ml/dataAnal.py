import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score, accuracy_score
from scipy.optimize import curve_fit


def clean_data():
    """
    clean the data and return a cleaned data frame
    """
    pd.set_option('display.max_columns', None)
    try:
        df = pd.read_csv('test1/movie.csv')
    except FileNotFoundError:
        df = pd.read_csv('movie.csv')

    df.drop(labels=["actor_3_facebook_likes", "actor_2_name",
                    "actor_1_facebook_likes", "actor_1_name",
                    "num_voted_users",
                    "cast_total_facebook_likes", "actor_3_name",
                    "facenumber_in_poster", "movie_imdb_link",
                    "num_user_for_reviews", "actor_2_facebook_likes",
                    "aspect_ratio", "color", "num_critic_for_reviews",
                    "director_facebook_likes"], axis=1, inplace=True)
    df.dropna(subset=["gross"], axis=0, inplace=True)
    return df


'''
predict some other thing
'''


def other():
    """
    test some other ML diagram here
    """
    df = clean_data()
    # splits each item in the column and returns the first split item
    df["genres"] = df["genres"].apply(lambda x: x.split("|")[0])
    print(df.tail())


'''
Predict the gross of a movie based on the imdb score given for that movie
'''


def func(x, a, b, c):
    return a * np.exp(b * x) + c


class gross_by_score:

    def __init__(self):
        """
        initialize the df
        """
        self.df = clean_data()

    def process_data(self):
        """
        take in the csv, clean the data, fit the data to a model and then return the parameters
        :return:
        """
        y = self.df['gross']
        x = self.df['imdb_score']

        # plt.scatter(x, y, color='blue', label="data")
        # plt.xlabel("imdb_score")
        # plt.ylabel("gross")

        # need to fit an exponential data set
        popt, pcov = curve_fit(func, x, y)
        # popt is parameters
        # X = np.arange(0.0, 10.0, 0.1)
        # plt.plot(X, func(X, popt[0], popt[1], popt[2]), 'r-', label="fit")
        # plt.legend(loc="best")
        # plt.show()

        # metrics.accuracy_score for accuracy
        acc = r2_score(y, func(x, popt[0], popt[1], popt[2]))

        return {"param": popt, "acc": acc}

    def prediction(self, score: float):
        """
        gets the required parameters and returns the predicted gross value
        according to this
        :return:
        """
        data = self.process_data()
        paramters = data["param"]
        return {"pred": func(score, paramters[0], paramters[1], paramters[2]),
                "acc": data["acc"]}


if __name__ == "__main__":
    other()

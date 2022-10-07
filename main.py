import pandas as pd
from scipy import stats
from statsmodels import robust
from pathlib import Path
import matplotlib.pyplot as plt


def func():
    data = pd.read_csv("googleplaystore.csv")
    columns = ["Size", "Current Ver"]
    for i in columns:
        df = data[data[i].str.contains("Varies with device") == False]
    df = df.dropna()
    df = df.drop_duplicates()

    filepath = Path('C:/Users/sasha/PycharmProjects/MTAD/googleplaystore_edit.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath)

    df = df.astype({'Reviews': 'int'})
    df['Size'] = df['Size'].apply(value_to_float)
    df['Last Updated'] = df['Last Updated'].apply(pd.to_datetime)
    df_mean = df[['Rating', 'Reviews', 'Size']].mean()

    file = open("result.txt", "w")

    file.write("Mean:\n")
    file.write(df_mean.to_string())
    file.write("\n\n")

    file.write("Trimmed Mean(10%):\n")
    file.write("Rating\t" + str(stats.trim_mean(df.Rating, 0.1)) + "\n")
    file.write("Reviews\t" + str(stats.trim_mean(df.Reviews, 0.1)) + "\n")
    file.write("Size\t" + str(stats.trim_mean(df.Size, 0.1)) + "\n")
    file.write("\n")

    file.write("Median:\n")
    file.write(df[['Rating', 'Reviews', 'Size']].median().to_string())
    file.write("\n\n")

    file.write("Variance:\n")
    file.write(df[['Rating', 'Reviews', 'Size']].var().to_string())
    file.write("\n\n")

    file.write("Standard Deviation:\n")
    file.write(df[['Rating', 'Reviews', 'Size']].std().to_string())
    file.write("\n\n")

    file.write("Mean Absolute Deviation:\n")
    file.write((df[['Rating', 'Reviews', 'Size']] - df_mean).abs().mean().to_string())
    file.write("\n\n")

    file.write("Median Absolute Deviation:\n")
    file.write(df[['Rating', 'Reviews', 'Size']].apply(robust.mad).to_string())

    file.close()

    columns = ['Rating', 'Reviews', 'Size']
    filepath = Path('C:/Users/sasha/PycharmProjects/MTAD/MinMaxNorm.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    normalize_min_max(df, columns).to_csv(filepath)
    filepath = Path('C:/Users/sasha/PycharmProjects/MTAD/MeanNorm.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mean_normalize(df, columns).to_csv(filepath)

    # df.head(25).plot(y='App', x='Reviews', kind='scatter', color='red')
    # plt.show()
    # df.head(1000).groupby('Category')['App'].nunique().plot(kind='bar')
    # plt.show()


def value_to_float(x):
    if type(x) == float or type(x) == int:
        return x
    if 'K' in x:
        if len(x) > 1:
            return float(x.replace('K', '')) * 1000
        return 1000.0
    if 'M' in x:
        if len(x) > 1:
            return float(x.replace('M', '')) * 1000000
        return 1000000.0
    return 0.0


def normalize_min_max(df_, columns):
    df = df_.copy()
    for column in columns:
        df[column] = ((df[column] - df[column].min()) / (df[column].max() - df[column].min())).round(3)
    return df


def mean_normalize(df_, columns):
    df = df_.copy()
    for column in columns:
        df[column] = ((df[column] - df[column].mean()) / df[column].std()).round(3)
    return df


if __name__ == '__main__':
    func()

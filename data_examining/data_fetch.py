import pandas as pd



# get data from file
def fetch_data(filename):
    if '.csv' in filename:
        return pd.read_csv(filename)
    elif '.txt' in filename:
        return pd.read_table(filename)
    elif '.xlsx' in filename:
        return pd.read_excel(filename)
    else:
        return 'file type not supported'


if __name__ == '__main__':
    a = fetch_data('../source_datasets/supermarket.csv')
    print(a.columns.values.tolist())
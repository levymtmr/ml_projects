
import os
import pandas
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

HOUSING_PATH = "./datasets/"

def load_housing_data(housing_data=HOUSING_PATH):
    csv_path = os.path.join(housing_data, 'housing.csv')
    return pandas.read_csv(csv_path)

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_ : test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing = load_housing_data()
housing.hist(bins=50, figsize=(20, 15))
plt.show()

# generate number to get sample of data randomly
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print(len(train_set))
# print(len(test_set))




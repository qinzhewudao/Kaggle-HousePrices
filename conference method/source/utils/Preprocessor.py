import numpy as np
import pandas as pd


class Preprocessor(object):
    def __init__(self):
        pass

    def preprocess(self, train, test, except_num=False):
        all_data = pd.concat((train.drop('SalePrice', axis=1), test), axis=0)
        all_data = all_data.fillna(all_data.mean()) # fill NA and Nan

        # delete columns which have number type
        if except_num:
            for column in all_data.columns:
                dtype = all_data[column].dtype
                if dtype in ['float64', 'int64']:
                    all_data = all_data.drop(column, axis=1)
        else:
            pass

        all_data = pd.get_dummies(all_data)
        all_data = all_data.fillna(all_data.mean())
        train = all_data[:train.shape[0]]
        test = all_data[train.shape[0]:]
        return train, test

if __name__ == '__main__':
    from CSVHandler import CSVHandler
    csv_handler = CSVHandler('../../data')
    preprocessor = Preprocessor()
    train = csv_handler.load_csv('train.csv')
    test = csv_handler.load_csv('test.csv')
    train, test = preprocessor.preprocess(train, test)
    print(train, test)

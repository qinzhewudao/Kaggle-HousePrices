import csv
import pandas as pd
import os

class CSVHandler(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_csv(self, filename):
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath, index_col='Id')

    def save_csv(self, output, model_name):
        predictions_file = open('predict/{0}_output.csv'.format(model_name), 'w')
        open_file_object = csv.writer(predictions_file)
        open_file_object.writerow(['Id','SalePrice'])
        open_file_object.writerows(output)
        predictions_file.close()

if __name__ == '__main__':
    csv_handler = CSVHandler('../../data')
    train = csv_handler.load_csv('train.csv')
    print(train.query('index == 1'))

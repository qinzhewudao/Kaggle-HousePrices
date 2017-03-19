import sys
import os

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV

from utils.CSVHandler import CSVHandler
from utils.Preprocessor import Preprocessor
# from utils.Visualizer import Visualizer
from utils.Logger import Logger


"""
File paths
"""
data_dir = '../data/'
train_filename = 'train.csv'
test_filename = 'test.csv'

"""
Make directory to save result
"""
try:
    os.mkdir('./predict/')
except:
    pass


# main loop
def main():
    csv_handler = CSVHandler(data_dir)
    preprocessor = Preprocessor()
    # visualizer = Visualizer()
    logger = Logger()

    # print "load train data and test data"
    try:
        train = csv_handler.load_csv(train_filename)
        test = csv_handler.load_csv(test_filename)
    except Exception as e:
        logger.show_exception(e)

    # print "preprocess the both data"
    t_train = train["SalePrice"].values
    train, test = preprocessor.preprocess(train, test, except_num=True)

    # print "extract target column and feature column for both data"
    x_train = train.values
    x_test = test.values

    # print "save test ids"
    test_ids = test.index

    # print "design training"
    tuned_parameters = [{'C': [1000, 10000, 100000], 'epsilon': [1000, 100, 10]}]
    reg = GridSearchCV(
        SVR(),
        tuned_parameters,
        cv=5
    )

    # print "train"
    reg.fit(x_train, t_train)
    logger.show_training_result(reg)

    # print "prediction"
    y_train = reg.predict(x_train).astype(int)
    y_test = reg.predict(x_test).astype(int)

    # print "save"
    output = zip(test_ids, y_test)
    csv_handler.save_csv(output, 'support_vector_regression')

    # print "show difference between true distribution and prediction"
    # visualizer.show_result(t_train, y_train)

    # print "everything works well"
    return 0


if __name__ == '__main__':
    sys.exit(main())

import pandas as pd

import matplotlib
from matplotlib import pyplot as plt


class Visualizer(object):
    def __init__(self):
        pass

    def show_result(self, t, y):
        matplotlib.rcParams['figure.figsize'] = (6.0, 4.0)
        prices = pd.DataFrame({'true distribution': t, 'prediction': y})
        ax= prices.plot(bins=50, alpha=0.5, figsize=(10,6), kind='hist')
        plt.show()

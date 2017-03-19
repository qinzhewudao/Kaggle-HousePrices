import sys


class Logger(object):
    def __init__(self):
        pass

    def show_training_result(self, model):
        for params, mean_score, all_scores in model.grid_scores_:
            print("{0:.3f} (+/- {1:.3f}) for {2}".format(mean_score, all_scores.std() / 2, params))
        print('best parameter:{0}'.format(model.best_params_))

    def show_exception(self, e):
        print("Exception:")
        print('  type     -> ', str(type(e)))
        print('  args     -> ', str(e.args))
        print('  message  -> ', e.message)
        print('  e        -> ', str(e))
        sys.exit(1)

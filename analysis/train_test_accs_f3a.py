import utils.constants as c
import utils.helper as h
import utils.analyses as a
import utils.plotting as p

import pandas as pd
import numpy as np
from scipy.stats import binom

# which participants to include
participants = [1,2,7,8,9,10,12] # those who learn | see paper Section 3.1

# get ci bounds for chance level
ci_bounds = binom.interval(.95, 36, (1/3))
ci_lower = ci_bounds[0] / 36
ci_upper = ci_bounds[1] / 36

# Get data human data
human_data = pd.read_csv(c.PATH_DATA + 'human_classification_data.csv')
# get only the participants we want
human_data = human_data[human_data['run'].isin(participants)]
# get the training and testing mean accuracy per participant and epoch
human_train_sum, human_test_sum = a.accuracy_per_run_an_epoch(human_data)
# get human mean and sd per epoch
human_train_mean_sd, human_test_mean_sd = a.mean_accuracy_per_epoch(human_train_sum,
                                                                    human_test_sum)

#loop over models
for model in c.MODELS:
    model_data = pd.read_csv(c.PATH_DATA + f'{model}_classification_data.csv')

    # get the training and testing mean accuracy per run and epoch
    model_train_sum, model_test_sum = a.accuracy_per_run_an_epoch(model_data)
    # get model mean and sd per epoch
    model_train_mean_sd, model_test_mean_sd = a.mean_accuracy_per_epoch(model_train_sum,
                                                                        model_test_sum)
    
    print(model)
    print('model_test_mean_sd', model_test_mean_sd)

    #plot train and test accuracy
    p.plot_performance(c.PATH_PLOTS + 'performance/' + 'performance_' + model + '.png',
                           model,
                           human_train_mean_sd['Train'], human_test_mean_sd['Test'],
                           model_train = model_train_mean_sd['Train'],
                           model_test = model_test_mean_sd['Test'],
                           titel=c.MODELS_DICT[model],
                            ci_lower=ci_lower, ci_upper=ci_upper)

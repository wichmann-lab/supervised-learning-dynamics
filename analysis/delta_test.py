import pandas as pd
import numpy as np
from scipy.stats import binom


import utils.constants as c
import utils.analyses as a
import utils.plotting as p

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
human_train_mean_sd, human_test_mean_sd = a.mean_accuracy_per_epoch(human_train_sum, human_test_sum)

# at which epochs do humans reach above chance level?:
j_humans = a.get_first_above_chance_epoch(human_train_mean_sd, ci_upper)
print('humans first above chance epoch:', j_humans)

delta_test_dict = {}
dict_epochs = {}

#loop over models
for model in c.MODELS:
    model_data = pd.read_csv(c.PATH_DATA + f'{model}_classification_data.csv')

    # get the training and testing mean accuracy per run and epoch
    model_train_sum, model_test_sum = a.accuracy_per_run_an_epoch(model_data)

    # get human mean and sd per epoch
    model_train_mean_sd, model_test_mean_sd = a.mean_accuracy_per_epoch(model_train_sum, model_test_sum)

    #get first epoch above chance and epoch where test accuracy decreases
    j = a.get_first_above_chance_epoch(model_train_mean_sd, ci_upper)
    k = a.get_epoch_test_accuracy_decreases(model_test_mean_sd)

    # from models_test_mean, get the accuracy from the first epoch above chance to the epoch where test accuracy decreases
    model_accs = model_test_mean_sd['Test'][(j-1):k]
    human_acss = human_test_mean_sd['Test'][(j-1):k]

    # add the epochs to the dict
    dict_epochs[model] = list(range(j, k+1))

    #get the difference between model and human accuracy
    diff = human_acss - model_accs

    # put the difference in a list
    diff = diff.tolist()
    # add the list to the dict
    delta_test_dict[model] = diff

# for key, value in delta_test_dict.items():
#     print(key, value.round(3))

p.plot_delta_test(delta_test_dict, dict_epochs)



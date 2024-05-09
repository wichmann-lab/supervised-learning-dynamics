import pandas as pd
import numpy as np
from scipy.stats import binom

import utils.constants as c
import utils.analyses as a
import utils.plotting as p

# which participants to include
participants = [1,2,7,8,9,10,12] # those who learn | see paper Section 3.1

# size of training set
n_train = 36

# initialize dict to store the results
efficiency_dict = {}

for model in c.MODELS:
    model_data = pd.read_csv(c.PATH_DATA + f'{model}_classification_data.csv')

    # get the training and testing mean accuracy per run and epoch
    model_train_sum, model_test_sum = a.accuracy_per_run_an_epoch(model_data)

    # get human mean and sd per epoch
    _, model_test_mean_sd = a.mean_accuracy_per_epoch(model_train_sum, model_test_sum)


    # add epoch 0 with Test 0 Test_sd 0 at the beginning of the dataframe
    model_test_mean_sd = pd.concat([pd.DataFrame({'Test': [0], 'Test_sd': [0]}), model_test_mean_sd], ignore_index=True)


    # for each epoch starting from 1, calculate the efficiency that is the difference between the current test accuracy and the previous test accuracy divided by the number of training samples
    model_test_mean_sd['Efficiency'] = (model_test_mean_sd['Test'].diff() / n_train).shift(-1)

    # add efficiency to the dict
    efficiency_dict[model] = model_test_mean_sd['Efficiency'].tolist()

# Get data human data
human_data = pd.read_csv(c.PATH_DATA + 'human_classification_data.csv')
# get only the participants we want
human_data = human_data[human_data['run'].isin(participants)]
# get the training and testing mean accuracy per participant and epoch
human_train_sum, human_test_sum = a.accuracy_per_run_an_epoch(human_data)
# get human mean and sd per epoch
_, human_test_mean_sd = a.mean_accuracy_per_epoch(human_train_sum, human_test_sum)

# add epoch 0 with Test 0 Test_sd 0 at the beginning of the dataframe
human_test_mean_sd = pd.concat([pd.DataFrame({'Test': [0], 'Test_sd': [0]}), human_test_mean_sd], ignore_index=True)

# for each epoch starting from 1, calculate the efficiency that is the difference between the current test accuracy and the previous test accuracy divided by the number of training samples
human_test_mean_sd['Efficiency'] = (human_test_mean_sd['Test'].diff() / n_train).shift(-1)

# add efficiency to the dict
efficiency_dict['human'] = human_test_mean_sd['Efficiency'].tolist()

for key, value in efficiency_dict.items():
    print(key, value)

p.plot_data_efficiency(efficiency_dict)



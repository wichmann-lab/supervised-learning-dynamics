import utils.constants as c
import utils.helper as h
import utils.analyses as a
import utils.plotting as p

from scipy.stats import binom
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Get data human data
human_data = pd.read_csv(c.PATH_DATA + 'human_classification_data.csv')


# Info for confidence interval
# Chance level
chance_level = 1/3

# Confidence level (e.g., 95%)
confidence_level = 0.95

for phase in ['train', 'test']:

    # List of window sizes for moving average and n for chance confidence interval
    if phase == 'train':
        window_sizes = [9, 18, 36]
        n_conf_int = 9
        epoch_length = 36
        ci = 'observer_mean'
        #only get phase data
        data = human_data.copy()
        data = data[data['phase'] == phase]
        
    elif phase == 'test':
        window_sizes = [13, 26, 51]
        n_conf_int = 51
        epoch_length = 51
        ci = 'observer_mean'
        #only get phase data
        data = human_data.copy()
        data = data[data['phase'] == phase]

    else:
        raise ValueError('data_set must be either "train" or "test"')
    # sort by run and epoch
    data = data.sort_values(by=['run', 'epoch'])

    # get a list of df for each run
    participants_data = []
    for run in data['run'].unique():
        participants_data.append(data[data['run'] == run])

    p.plot_moving_average(phase, participants_data,
                        window_sizes, epoch_length,
                        ci=None, confidence_level=None, n_conf_int=None,chance_level=None)
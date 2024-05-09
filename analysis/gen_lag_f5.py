import pandas as pd
from scipy.stats import binom


import utils.constants as c
import utils.analyses as a
import utils.plotting as p

# models data dict (acc is imagenet accuracy from pytorch hub)
models_data = {
    'Humans': {'acc': 0.8303, 'lag': None,'epochs': None, 'param': None, 'color': '#dd3497', 'marker': 's', 'alpha': 1},
    'ViT\n(86.6M)': {'acc': 0.81072, 'lag': None,'epochs': None, 'param': 86.6, 'color': 'orange', 'marker': 'o', 'alpha': 0.6},
    'EfficientNet\n(54.1M)': {'acc': 0.85112, 'lag': None,'epochs': None, 'param': 54.1, 'color': 'orange', 'marker': 'o', 'alpha': 0.6},
    'ConvNeXt\n(88.6M)': {'acc': 0.84062, 'lag': None,'epochs': None, 'param': 88.6, 'color': 'orange', 'marker': 'o', 'alpha': 0.6},
    'AlexNet\n(81.1M)': {'acc': 0.56522, 'lag': None,'epochs': None, 'param': 61.1, 'color': 'teal', 'marker': 'o', 'alpha': 0.6},
    'ResNet-50\n(25.6M)': {'acc': 0.7613, 'lag': None,'epochs': None, 'param': 25.6, 'color': 'teal', 'marker': 'o', 'alpha': 0.6},
    'VGG-16\n(138.4M)': {'acc': 0.71592, 'lag': None,'epochs': None, 'param': 138.4, 'color': 'teal', 'marker': 'o', 'alpha': 0.6}
}

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


# get human generalisation lag pooled over participants:
j = a.get_first_above_chance_epoch(human_train_mean_sd, ci_upper)
gen_lag_humans = ((human_train_mean_sd['Train'][(j-1):] - human_test_mean_sd['Test'][(j-1):]).sum())/(7-j)

# add gen_lag and epochs above chance (j) to models_data humans dict key 'lag' and key 'epochs'
models_data['Humans']['lag'] = gen_lag_humans
models_data['Humans']['epochs'] = j

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


    print(model)
    print('test accs\n', model_test_mean_sd['Test'][(j-1):k])
    print('devided by ', (len(range((j-1),k))))

    # calculate generalisation lag
    gen_lag_model = ((model_train_mean_sd['Train'][(j-1):k] - model_test_mean_sd['Test'][(j-1):k]).sum())/(len(range((j-1),k)))

        # add gen_lag and epochs above chance (j) to models_data humans dict key 'lag' and key 'epochs'
    models_data[c.MODELS_DICT_PARAM[model]]['lag'] = gen_lag_model
    models_data[c.MODELS_DICT_PARAM[model]]['epochs'] = (len(range((j-1),k)))

p.plot_gen_lag(models_data, poster=False)

#print each key and value in models_data

for key, value in models_data.items():
    print(key, value)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import utils.constants as c
import utils.analyses as a
import utils.plotting as p

#loop over models
for model in c.MODELS:
    print(model)

    #get color
    if model in c.CLASSIC_MODELS_DICT:
        color = c.CLR_MODELS[1]
    elif model in c.SOTA_MODELS_DICT:
        color = c.CLR_MODELS[0]

    #load data
    model_data = pd.read_csv(c.PATH_DATA + f'{model}_classification_data.csv')
    model_data_train, model_data_test = a.accuracy_per_run_an_epoch(model_data)

    #get means over runs per epoch
    model_data_train_means, model_data_test_means = a.mean_accuracy_per_epoch(model_data_train, model_data_test)



    #plot train and test accuracy of single runs
    p.plot_model_runs(model_data_train, model_data_test,
                    model_data_train_means, model_data_test_means,
                    model, color)

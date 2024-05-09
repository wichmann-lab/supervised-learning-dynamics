import os
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

import utils.constants as c
import utils.helper as h
import utils.analyses as a
import utils.plotting as p

data_path = c.PATH_DATA
data_files = h.find_csv_files(data_path)

confusion_dict_all = {}

# which participants to include
participants = [1,2,7,8,9,10,12] # those who learn | see paper Section 3.1

for file in data_files:
    df = pd.read_csv(file)
    confusion_dict_phases = {}
    for phase in ['train', 'test']:

        #only get phase data
        df_phase = df.copy()
        df_phase = df_phase[df_phase['phase'] == phase]
  

        subject = file.split('/')[-1].split('_')[0]
        data_type = phase
        if subject == 'human':
            df_phase = df_phase[df_phase['run'].isin(participants)]

        df_phase = df_phase.rename(columns={'ground_truth': 'category'})
        df_phase = df_phase.rename(columns={'prediction': 'category_response'})

        if data_type == 'train':
            df_phase['epoch'] = 'learn_train_' + df_phase['epoch'].astype(str)
        elif data_type == 'test':
            df_phase['epoch'] = 'learn_test_' + df_phase['epoch'].astype(str)

        df_phase = df_phase.rename(columns={'epoch': 'trial_type'})

        cm_dict = p.plot_confusion_matrix_overall(c.PATH_PLOTS + 'confusion/' + subject + '_' + data_type + '_confusion.png',
                                        df_phase)

        confusion_dict_phases[data_type] = cm_dict
    confusion_dict_all[subject] = confusion_dict_phases


# only get the confusion matrix for the test phase
confusion_dict_test = {}
for subject in confusion_dict_all.keys():
    confusion_dict_test[subject] = confusion_dict_all[subject]['test']

for epoch in ['learn_test_1', 'learn_test_2', 'learn_test_3', 'learn_test_4', 'learn_test_5', 'learn_test_6']:

    # from confusion_dict_test, only get the confusion matrix for the learn_test_1 epoch for all
    confusion_dict_learn_test_1 = {}
    for subject in confusion_dict_test.keys():
        confusion_dict_learn_test_1[subject] = confusion_dict_test[subject][epoch]


    # List of model names and number of models
    models = list(confusion_dict_learn_test_1.keys())
    n_models = len(models)

    # Initialize the distance matrix
    distance_matrix = np.zeros((n_models, n_models))

    # Calculate pairwise Manhattan distances
    for i in range(n_models):
        for j in range(i + 1, n_models):
            distance = np.sum(np.abs(confusion_dict_learn_test_1[models[i]] - confusion_dict_learn_test_1[models[j]]))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix

    # Convert the distance matrix to a numpy array if it's not already
    distance_matrix = np.array(distance_matrix)

    # Perform MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_coords = mds.fit_transform(distance_matrix)

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.scatter(mds_coords[:, 0], mds_coords[:, 1], color='blue', label='Models')
    for label, x, y in zip(models, mds_coords[:, 0], mds_coords[:, 1]):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    plt.grid(True)
    plt.title('MDS Embedding of the Models Based on Confusion Matrix Distances')
    plt.xlabel('MDS Dimension 1')
    plt.ylabel('MDS Dimension 2')
    plt.legend()
    plt.savefig(c.PATH_PLOTS + f'mds_confusion/{epoch}_mds_confusion.png', dpi=300)
    plt.close()

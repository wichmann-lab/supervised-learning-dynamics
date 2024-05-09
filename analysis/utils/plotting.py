import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import confusion_matrix
from scipy.stats import binom
import random


import utils.helper as h
import utils.analyses as a
import utils.constants as c

def plot_performance(path, model_name,
                         human_train, human_test,
                         model_train, model_test,
                         titel=None,
                         ci_lower=None, ci_upper = None):
    
    # get colors
    if model_name in c.CLASSIC_MODELS_DICT.keys():
        color = c.CLR_MODELS[1]
        tag = 'Classic\nmodels'
    elif model_name in c.SOTA_MODELS_DICT.keys():
        color = c.CLR_MODELS[0]
        tag = 'SOTA\nmodels'
    else:
        print('Model name not found')

    if model_name != 'resnet50' and model_name != 'vit':
        frameon = False
        draw_legend = False
    else:
        frameon = True
        draw_legend = True
    
    fig, ax = plt.subplots(figsize=(5, 4))
    # Create dummy lines for legend
    line_train = mlines.Line2D([], [], color='grey', linestyle='--', linewidth=2.5,  label='Train')
    line_test = mlines.Line2D([], [], color='grey', linestyle='-',linewidth=2.5, label='Test')
    line_chance = mlines.Line2D([], [], color='grey', linestyle=':', linewidth=1.5, label='Chance')
    marker_human = mlines.Line2D([], [], color='#dd3497', marker='s', linestyle='None',markersize=10, label='Humans')
    marker_model = mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=tag)

    df_human = pd.DataFrame({'Train': human_train, 'Test': human_test})
    df_model = pd.DataFrame({'Train': model_train, 'Test': model_test})
    offset_x_values = df_human.index

    #plot chance performance and chance confidence interval
    plt.axhline(y=0.3333333, color='gray', linestyle=':', linewidth=1.5)
    # Plot confidence interval bounds
    if ci_lower is not None and ci_upper is not None:

        plt.axhspan(ci_lower, ci_upper, color='grey', alpha=0.2, lw=0)

    #plot human train and test performance
    sns.lineplot(x=df_human.index, y=df_human['Train'], color='#dd3497', linestyle ='--', linewidth=2.5)
    sns.lineplot(x=offset_x_values, y=df_human['Test'], color='#dd3497', linewidth=2.5)
    sns.scatterplot(x=offset_x_values, y=df_human['Test'], marker='s', color='#dd3497', label = 'Test', s=80)
    sns.scatterplot(x=df_human.index, y=df_human['Train'], marker='s', color='#dd3497', label = 'Train', s=80)



    #plot model train and test performance #ffc966 #66b2b2
    sns.lineplot(x=df_model.index, y=df_model['Train'], color=color, linestyle ='--', linewidth=2.5)
    sns.lineplot(x=offset_x_values, y=df_model['Test'], color=color, linestyle ='-', linewidth=2.5)
    sns.scatterplot(x=offset_x_values, y=df_model['Test'], marker='o', color=color, label = 'Test', s=80)
    sns.scatterplot(x=df_model.index, y=df_model['Train'], marker='o', color=color, label = 'Train', s=80)

    plt.title(titel, fontsize=18)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    plt.ylim(0, 1.05) # specify the range of the y-axis
    plt.xticks(np.arange(0, len(human_train), 1.0)) #x-axis only integers
    plt.xticks(np.arange(0, len(human_train), 1.0), np.arange(1, len(human_train)+1, 1), fontsize=16) # specify the x-ticks labels
    plt.yticks(fontsize=16)

    # Get current axis
    ax = plt.gca()

    # Set the width of the axes lines
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.tick_params(axis='both', which='major', width=2)  # For major ticks

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    # plt.legend(by_label.values(), by_label.keys(), markerscale=1,
    #            loc='lower left', frameon=False)

    legend = plt.legend(handles=[line_train, line_test, line_chance, marker_human, marker_model],
               ncol=2, loc=(0.225, 0.1), fontsize=14, frameon=frameon)
    
    if not draw_legend:
        for text in legend.get_texts():
            text.set_alpha(0)
        for handle in legend.legendHandles:
            handle.set_alpha(0)

    #plt.legend().remove()
    plt.tight_layout()
    sns.despine(trim=True)
    plt.grid(False)
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_test_performance_novel_known(path,
                                      accs_known, accs_novel,
                                      model):

    title = c.MODELS_DICT[model]

    # X-axis values (1 to 6)
    x_values = range(1, 7)

    # Create a figure and an axes.
    fig, ax = plt.subplots(figsize=(2.8, 2.2))
    
    # Plotting both lists
    ax.plot(x_values, accs_known, label='Novel views',
            linestyle='-', marker='o', color = 'gray',
            linewidth=1.8) 
    ax.plot(x_values, accs_novel, label='Novel objects',
            linestyle='--', marker='o', color = 'gray',
            linewidth=1.8)  # Dashed line with x markers

    # Create custom handles for the legend
    legend_handles = [
        Line2D([0], [0], color='gray', linewidth=1.8, linestyle='--', label='Novel objects'),
        Line2D([0], [0], color='gray', linewidth=1.8, linestyle='-', label='Novel views')
    ]

    if model == 'human':
        frameon = True
    else:
        frameon = False

    # Adding legend to the plot
    legend = ax.legend(handles=legend_handles, loc = 'lower right', fontsize=13, frameon=frameon) 

    if model != 'human':
        for text in legend.get_texts():
            text.set_alpha(0)
        for handle in legend.legendHandles:
            handle.set_alpha(0)

    # Set y-axis limits to 0-1
    ax.set_ylim(0, 1.08)

    # Add titles 
    ax.set_title(title)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    # # Set the width of the axes lines
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    plt.tick_params(axis='both', which='major', width=1.5)  # For major ticks

    # Adjusting the x-axis and y-axis
    ax.set_xticks([1, 2, 3, 4, 5, 6])
 
    ax.set_xlim(0.8, 6.2)
    plt.xticks(fontsize=14)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(-0.06, 1.06)
    plt.yticks(fontsize=14)

    # Cosmetics and save
    sns.despine(trim=True)
    #plt.show()
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_model_runs(train_data, test_data,
                    model_data_train_means, model_data_test_means,
                    model, color):


    # plot train and test accuracy
    for phase in ['Train', 'Test']:

        if phase == 'Train':
            linestyle = '--'
            model_data = train_data
            model_data_means = model_data_train_means
        else:
            linestyle = '-'
            model_data = test_data
            model_data_means = model_data_test_means
        # Setup figure and axis
        fig, ax = plt.subplots(figsize=(5, 3.5))

        #plot chance performance and chance confidence interval
        plt.axhline(y=0.3333333, color='gray', linestyle=':', linewidth=2)

        #plot individual runs
        for run in model_data['run'].unique():

            run_data = model_data[model_data['run'] == run]

            plt.plot(run_data['epoch']-1, run_data[f'{phase}'], label=f'Run {run} - Train',
                    color='gray', alpha=0.3, linestyle=linestyle)

        sns.lineplot(x=model_data_means['Epoch']-1, y=model_data_means[phase], color=color, linestyle =linestyle, linewidth=3, alpha=1)
        sns.scatterplot(x=model_data_means['Epoch']-1, y=model_data_means[phase], marker='o', color=color, label = phase, s=80, alpha=1)


        plt.legend().remove()
        plt.title(c.MODELS_DICT[model], fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.ylabel('Accuracy (top-1)', fontsize=18)
        plt.ylim(0, 1.05) # specify the range of the y-axis
        plt.xticks(np.arange(0, len(model_data_means), 1.0)) #x-axis only integers
        plt.xticks(np.arange(0, len(model_data_means), 1.0), np.arange(1, len(model_data_means)+1, 1), fontsize=16) # specify the x-ticks labels
        plt.yticks(fontsize=16)
        plt.tight_layout()
        sns.despine(trim=True)
        plt.grid(False)
        #plt.show()
        plt.savefig(c.PATH_PLOTS + f'model_runs/{model}_{phase}.png', dpi=300)
        plt.close()

def plot_gen_lag(dict, poster=False):

    if poster:
        fontsize = 14
        anotate_size = 11
    else:
        fontsize = 10
        anotate_size = 10

    # Setup figure and axis
    fig, ax = plt.subplots(figsize=(3.8, 5.2))

    # Plot each data point
    # Plot each data point
    for model, info in dict.items():
        size = info['param']*10 if info['param'] is not None else 100  # Circle area proportional to params
        if model != 'Humans':
            ax.scatter(info['acc'], info['lag'], s=size, color=info['color'], alpha=info['alpha'], label=model, marker=info['marker'])

    # Plot the line for humans
    human_lag = dict['Humans']['lag']
    ax.plot([0.804, 0.9], [human_lag, human_lag], color=dict['Humans']['color'], linewidth=5, label='Humans')

    plt.annotate("ViT\n(86.6M)", (0.822, .165), textcoords="offset points", xytext=(0,-16), size=anotate_size)
    plt.annotate('EfficientNet\n(54.1M)', (0.85, .268), textcoords="offset points", xytext=(0,-16), size=anotate_size)
    plt.annotate("ConvNeXt\n(88.6M)", (0.82, .058), textcoords="offset points", xytext=(0,-16), size=anotate_size)
    plt.annotate("AlexNet\n(81.1M)", (0.527, 0.11), textcoords="offset points", xytext=(0,-16), size=anotate_size)
    plt.annotate("ResNet-50\n(25.6M)", (0.64, .252), textcoords="offset points", xytext=(0,-16), size=anotate_size)
    plt.annotate("VGG-16\n(138.4M)", (0.665, .115), textcoords="offset points", xytext=(0,-16), size=anotate_size)
    plt.annotate("Humans", (0.8, .006), textcoords="offset points", xytext=(0,-16), size=anotate_size)

    if poster:
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        plt.tick_params(axis='both', which='major', width=1.2)  # For major ticks
        # Increase the size of the tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)


    # Set the x and y axis labels
    ax.set_xlabel('ImageNet Accuracy (top-1)', fontsize=fontsize)
    ax.set_ylabel('Generalisation Lag ($\Delta$G)', fontsize=fontsize)

    # Set the x and y axis limits
    ax.set_xlim(0.5, 0.9)
    ax.set_ylim(-0.025, 0.3)

    # Increase the size of the legend
    #ax.legend(loc='upper left', fontsize='large')

    plt.tight_layout()
    sns.despine(trim=True)
    plt.grid(False)
    plt.savefig(c.PATH_PLOTS + 'gen_lag/gen_lag_overview_poster.png', dpi=300)

def plot_confusion_matrix_overall(path, df, subject = None):
    # List of unique trial_types and categories
    trial_types = df['trial_type'].unique()
    categories = df['category'].unique()

    # Sort the categories to make sure the confusion matrix axes are consistent
    categories.sort()

    # Initialize the figure
    fig, axes = plt.subplots(1, len(trial_types), figsize=(15, 2.9))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Create a colormap
    cmap = sns.color_palette("PuRd", as_cmap=True)


    dict_cm = {}
    # Loop through each trial_type to plot
    for i, trial_type in enumerate(trial_types):
        ax = axes[i]
        subset_df = df[df['trial_type'] == trial_type]
        cm = confusion_matrix(subset_df['category'], subset_df['category_response'], labels=categories)
        # Normalize the confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap, xticklabels=categories, yticklabels=categories, ax=ax, vmin=0, vmax=1, cbar=i == len(trial_types)-1)
        title = re.search(r'(test_|train_)\d+', trial_type).group().replace('_', ' ')
        ax.set_title(f'{title}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        # Save the confusion matrix to a dictionary
        dict_cm[trial_type] = cm_normalized

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    return dict_cm

def plot_moving_average(phase, participants_data,
                        window_sizes, epoch_length,
                        ci=None, confidence_level=None, n_conf_int=None,chance_level=None):
 
    # Plotting setup for 2x3 subplots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 10))

    if ci == 'chance':
        # Calculate the confidence interval for the chance level
        ci_bounds = binom.interval(confidence_level, n_conf_int, chance_level)
        ci_lower = ci_bounds[0] / n_conf_int
        ci_upper = ci_bounds[1] / n_conf_int

    count = 0
    # Iterate through each participant's data
    for i, data in enumerate(participants_data):
        # Ensuring the DataFrame index represents trial numbers correctly
        data.reset_index(drop=True, inplace=True)
        ## add one to the index to start at 1
        data.index += 1

        # Calculate accuracy for each trial for this participant
        data['accuracy'] = data['ground_truth'] == data['prediction']

        # Calculate mean accuracy and number of trials for this participant
        mean_accuracy = data['accuracy'].mean()
        n_trials = len(data)
        
        if ci == 'observer_mean':
            # Calculate the confidence interval for the mean performance of this participant
            ci_bounds = binom.interval(confidence_level, n_trials, mean_accuracy)
            ci_lower = ci_bounds[0] / n_trials
            ci_upper = ci_bounds[1] / n_trials  

        # Select the subplot
        ax = axes[i // 3, i % 3]
        ax.set_title(f'Participant {data["run"].unique()[0]}')
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel('Accuracy')
        ax.grid(False)     

        if ci == 'observer_mean':
            # Plot confidence interval bounds
            ax.axhline(y=ci_lower, color='grey', linestyle='--')
            ax.axhline(y=ci_upper, color='grey', linestyle='--')

        # Calculate and plot moving average for each window size for this participant
        for i, window in enumerate(window_sizes):
            data[f'moving_avg_accuracy_{window}'] = data['accuracy'].rolling(window=window).mean()
            valid_range = data.index >= window - 1  # Start plotting when the window size is complete
            ax.plot(data.index[valid_range], data[f'moving_avg_accuracy_{window}'][valid_range],
                    marker=None, color=c.CLR_MOVING_AVRG[i], label=f'Window: {window}')
        # Adjusting the x-axis to start at 1 and continue in steps of 6 trials
        ax.set_xticks([0] + list(range(epoch_length, len(data) + 1, epoch_length)))
        ax.xlim = (1, len(data)+1)
        ax.set_xlim(-5, len(data)+1)
        ax.set_ylim(-0.02, 1.06)
        # Adding legend only for the first subplot
        if count == 0: 
            ax.legend()
        sns.despine(trim=True)
        count += 1

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjusting top to accommodate suptitle
    plt.savefig(c.PATH_PLOTS + f'moving_avrg/moving_avrg_{phase}.png', dpi=300)

def plot_delta_test(dict, dict_epochs):
    #get the mean of the lists in the dict in a new dict
    dict_mean = {}
    for model, diff in dict.items():
        mean_diff = np.mean(diff)
        dict_mean[model] = mean_diff

    # Plotting setup
    fig, ax = plt.subplots(figsize=(4, 2.5))

    ax.set_ylabel('Delta to Human\nTest Accuracy', fontsize=10)

    # add a line at 0
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)


    # Plot the mean difference in test accuracy for each model
    for model, diff in dict_mean.items():
        ax.bar(c.MODELS_DICT[model], diff,
            color=c.CLR_MODELS[1] if model in c.CLASSIC_MODELS_DICT else c.CLR_MODELS[0],
            width=0.8)  # Adjust the width parameter to make the bars thinner
    
    # Map model names to indices
    model_indices = {model: idx for idx, model in enumerate(c.MODELS_DICT.keys())}

    # Plot the entries in dict as individual points
    for model, diff in dict.items():
        #set random seed
        random.seed(42)
        #get the epoch numbers from the dict_epochs
        epochs = dict_epochs[model]
        for e, d in zip(epochs, diff):
            jitter = random.uniform(-0.13, 0.13)  # Add jitter to x-coordinate
            jitter_y = random.uniform(-0.015, 0.015)
            model_index = model_indices[model] + jitter  # Get numeric index and add jitter
            ax.plot([model_index+jitter], [d], 'o', color='gray', alpha=0.8, markersize=5)
            ax.text(model_index + jitter, d, str(e), ha='center', va='center', fontsize=5)


    # change the keys in model_indices to the full model names except humans
    model_indices = {c.MODELS_DICT[model]: idx for model, idx in model_indices.items() if model != 'human'}
    # Set x-axis ticks and labels
    ax.set_xticks(list(model_indices.values()))
    ax.set_xticklabels(list(model_indices.keys()))

    # Set the y-axis limits
    ax.set_ylim(-0.24, 0.15)

    # Adjust layout
    plt.xticks(rotation=45, fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    sns.despine(trim=True)
    plt.grid(False)
    plt.savefig(c.PATH_PLOTS + 'delta_test/delta_test.png', dpi=300)
    plt.close()
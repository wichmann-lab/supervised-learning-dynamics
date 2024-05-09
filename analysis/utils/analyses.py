import numpy as np

# Calculating accuracy per epoch for each run
def calculate_accuracy_np(run_data):
    return np.mean(run_data['ground_truth'] == run_data['prediction'])

def calc_accuracy(group):
    return (group['ground_truth'] == group['prediction']).mean()

def accuracy_per_run_an_epoch(data_df):
    df_train = data_df.copy()
    df_train = df_train[df_train['phase'] == 'train']
    df_test = data_df.copy()
    df_test = df_test[df_test['phase'] == 'test']

    # Grouping by run and epoch, then calculating mean and standard deviation of accuracy
    train_summary = df_train.groupby(['run', 'epoch']).apply(calculate_accuracy_np).reset_index()
    train_summary.columns = ['run', 'epoch', 'Train']

    # Grouping by run and epoch, then calculating mean and standard deviation of accuracy
    test_summary = df_test.groupby(['run', 'epoch']).apply(calculate_accuracy_np).reset_index()
    test_summary.columns = ['run', 'epoch', 'Test']

    return train_summary, test_summary

def mean_accuracy_per_epoch(train_summary, test_summary):
    # Calculating mean and standard deviation across runs for each epoch for train
    train_means_sd = train_summary.groupby('epoch')['Train'].agg(['mean', 'std']).reset_index()
    train_means_sd.columns = ['Epoch', 'Train', 'Train_sd']

    # Calculating mean and standard deviation across runs for each epoch for test
    test_means_sd = test_summary.groupby('epoch')['Test'].agg(['mean', 'std']).reset_index()
    test_means_sd.columns = ['Epoch', 'Test', 'Test_sd']

    return train_means_sd, test_means_sd

def analyse_test_performance (df, training_objects):

    novel_objects = df[df['image_name'].apply(lambda x: not any(x.startswith(object_name) for object_name in training_objects))]
    known_objects = df[df['image_name'].apply(lambda x: any(x.startswith(object_name) for object_name in training_objects))]
 

    result_all = df.groupby('epoch').apply(calc_accuracy).reset_index(name='accuracy')
    result_novel = novel_objects.groupby('epoch').apply(calc_accuracy).reset_index(name='accuracy')
    result_known = known_objects.groupby('epoch').apply(calc_accuracy).reset_index(name='accuracy')

    return result_novel['accuracy'].tolist(), result_known['accuracy'].tolist(), result_all['accuracy'].tolist()
    
def get_first_above_chance_epoch (summary, ci_upper):
    for i, row in summary.iterrows():
        if row['Train'] > ci_upper:
            j = row['Epoch'].astype(int)
            break
    return j

def get_epoch_test_accuracy_decreases(summary):
    # Initialize the previous test accuracy as infinity for the comparison in the first iteration
    prev_test_acc = 0
    
    # Initialize the epoch at which test accuracy starts decreasing
    decreasing_epoch = None
    
    # Iterate through each row in the summary data
    for i, row in summary.iterrows():
        # Get the current test accuracy
        current_test_acc = row['Test']
        
        # Check if the current test accuracy is less than the previous test accuracy
        if current_test_acc < prev_test_acc:
            # Test accuracy starts decreasing, mark the epoch
            decreasing_epoch = row['Epoch'].astype(int)
            break
        
        # Update the previous test accuracy for the next iteration
        prev_test_acc = current_test_acc
    if decreasing_epoch is not None:
        return decreasing_epoch-1
    else:
        return 6

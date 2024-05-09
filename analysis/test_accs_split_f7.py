import utils.constants as c
import utils.helper as h
import utils.analyses as a
import utils.plotting as p
import pandas as pd


def analyse_single_test_data_per_view ():

    data_path = c.PATH_DATA
    data_files = h.find_csv_files(data_path)

    for file in data_files:

        data = pd.read_csv(file)
        #remove the train data
        test_data = data[data['phase'] == 'test']

        # get obsevers name
        subject = file.split('/')[-1].split('_')[0]

        if subject == 'human':
            #filter the human data to get run 1,2,7,8,9,10,12
            test_data = test_data[test_data['run'].isin([1,2,7,8,9,10,12])]

        
        # analyse the performance (accuracy) per view
        accs_novel_test_obj, accs_known_test_obj, _ = a.analyse_test_performance(test_data, c.TRAINING_OBJECTS)

        p.plot_test_performance_novel_known(c.PATH_PLOTS + 'test_split/' + subject + '_test_performance_split_new.png',
                                            accs_known_test_obj, accs_novel_test_obj,
                                            subject)

if __name__ == "__main__":

    analyse_single_test_data_per_view()
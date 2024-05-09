import os
import pandas as pd

def dir_up(path,n): # here 'path' is your path, 'n' is number of dirs up you want to go
    for _ in range(n):
        path = dir_up(path.rpartition("/")[0], 0) 
    return(path)


def find_csv_files(directory_path):
    csv_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def add_row_to_dataframe(df, new_row):
    if df.empty:
        # If the DataFrame is empty, create a new DataFrame with the new row
        df = pd.DataFrame([new_row], columns=df.columns)
    else:
        # If the DataFrame already has rows, concatenate the new row to it
        new_df = pd.DataFrame([new_row], columns=df.columns)
        df = pd.concat([df, new_df], ignore_index=True)
    return df
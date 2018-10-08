# input: csv file with two columns, first for the X axis and second for the Y axis
# output: time series graph
import csv
import pandas as pd


def csv_to_dataframe(file):
    relative_path = "../data/"
    full_relative_path = relative_path + str(file)
    columns = []
    data = []
    with open(full_relative_path, 'rt', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for index, row in enumerate(csv_reader):
            if index == 0:
                columns = row
            else:
                data.append(row)

    df = pd.DataFrame(data, columns=columns)

    return df



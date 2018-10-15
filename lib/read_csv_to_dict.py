import csv

#read csv to list of dictionaries.
def csv_to_dict(csv_file):
    with open(csv_file) as f:
        dict_list = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]
    return dict_list
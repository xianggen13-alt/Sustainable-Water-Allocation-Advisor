import csv

# Path to your dataset
file_path = 'dataset/cleaned_global_water_consumption.csv'

# Open and read the CSV file
with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    # Print one of the headers
    print("One of the headers is:", reader.fieldnames[0])

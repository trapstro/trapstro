import pandas as pd

# Load the datasets
train_data = pd.read_csv(r"C:\Users\USER\Downloads\archive\ptbdataset\ptb.train.txt", sep='\t', header=None)
valid_data = pd.read_csv(r"C:\Users\USER\Downloads\archive\ptbdataset\ptb.valid.txt", sep='\t', header=None)
test_data = pd.read_csv(r"C:\Users\USER\Downloads\archive\ptbdataset\ptb.test.txt", sep='\t', header=None)

# Check the loaded data
print("Train Data:", train_data.head())
print("Validation Data:", valid_data.head())
print("Test Data:", test_data.head())

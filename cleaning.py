import pandas as pd

# Load the uploaded CSV file
file_path = '/mnt/data/Sample_Game_1_RawTrackingData_Away_Team.csv'
tracking_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
tracking_data.head()
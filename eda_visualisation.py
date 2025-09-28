# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV dataset
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
df = pd.read_csv(URL)

# Extract year from Date column
df['Year'] = pd.to_datetime(df['Date']).dt.year

# Select features for modeling
features = df[['FlightNumber', 'PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 
               'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial']]

# One-hot encode categorical columns
categorical_cols = ['Orbit', 'LaunchSite', 'LandingPad', 'Serial']
features_one_hot = pd.get_dummies(features, columns=categorical_cols)

# Print total number of columns after one-hot encoding
print("Total number of columns after one-hot encoding:", features_one_hot.shape[1])

# Save the processed dataset
features_one_hot.to_csv('dataset_part_3.csv', index=False)


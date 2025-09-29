# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd

# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices,
# along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np

# Matplotlib for visualization
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv")

# Display first 10 rows
print("First 10 rows of dataset:")
print(df.head(10))

# Check missing values (% of total)
print("\nPercentage of missing values in each column:")
print(df.isnull().sum()/len(df)*100)

# Check data types
print("\nData types of each column:")
print(df.dtypes)

# Number of launches per site
print("\nNumber of launches per Launch Site:")
launch_counts = df['LaunchSite'].value_counts()
print(launch_counts)

# Calculate number of launches per orbit
orbit_counts = df['Orbit'].value_counts()

# Exclude GTO since it's a transfer orbit
orbit_counts = orbit_counts.drop('GTO', errors='ignore')

# Print counts
print("\nNumber of launches per orbit (excluding GTO):")
print(orbit_counts)

# Calculate the number and occurrence of mission outcomes
landing_outcomes = df['Outcome'].value_counts()

# Print results
print("Number and occurrence of mission outcomes:")
print(landing_outcomes)

# Display the index and names of all unique outcomes
for i, outcome in enumerate(landing_outcomes.keys()):
    print(i, outcome)

# Define the set of "bad outcomes" (unsuccessful landings)
bad_outcomes = set(landing_outcomes.keys()[[1,3,5,6,7]])
print("\nBad outcomes (unsuccessful landings):")
print(bad_outcomes)

# Create landing_class: 0 if bad outcome, 1 otherwise
landing_class = [0 if outcome in bad_outcomes else 1 for outcome in df['Outcome']]

# Optional: show first 10 values
print("\nFirst 10 landing_class values:")
print(landing_class[:10])

df['Class']=landing_class
df[['Class']].head(8)
df.head(5)
df["Class"].mean()
df.to_csv("dataset_part_2.csv", index=False)

# Convert landing_class to a NumPy array for easier calculations
landing_class_array = np.array(landing_class)

# Calculate success rate
success_rate = landing_class_array.sum() / len(landing_class_array)

# Print success rate as a percentage
print(f"Success rate: {success_rate*100:.2f}%")

# Count the number of launches to GEO
geo_launches = df[df['Orbit'] == 'GEO'].shape[0]

print(f"Number of launches to Geosynchronous Orbit (GEO): {geo_launches}")

# Count number of successful landings on a drone ship
success_asds = df[df['Outcome'] == 'True ASDS'].shape[0]

print(f"Number of missions successfully landed on a drone ship (ASDS): {success_asds}")

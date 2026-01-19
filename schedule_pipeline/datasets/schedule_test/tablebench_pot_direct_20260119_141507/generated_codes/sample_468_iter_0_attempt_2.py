import pandas as pd

df = pd.read_csv('table.csv')

# Calculate mean and std of local magnitude
mean_mag = df['local magnitude'].mean()
std_mag = df['local magnitude'].std()

# Define threshold for outliers (2 standard deviations from mean)
threshold = 2 * std_mag
outliers = df[(df['local magnitude'] > mean_mag + threshold) | (df['local magnitude'] < mean_mag - threshold)]

print("Final Answer:", outliers.to_dict('records'))
import pandas as pd

df = pd.read_csv('table.csv')

# Filter female animals
female_data = df[df['Sex'] == 'Female']

# Convert 'Metabolic rate' to float
female_data['Metabolic rate'] = pd.to_numeric(female_data['Metabolic rate'])

# Mean is given as 1285.5
mean_metabolic_rate = 1285.5

# Calculate standard deviation
std_dev = female_data['Metabolic rate'].std()

# Find animals with metabolic rate more than one standard deviation away from the mean
outside_one_std = female_data[
    (female_data['Metabolic rate'] > mean_metabolic_rate + std_dev) |
    (female_data['Metabolic rate'] < mean_metabolic_rate - std_dev)
]

# Get the animal IDs (first column) of those outside one standard deviation
animals_outside = outside_one_std['Animal'].tolist()

# Output results
print(f"Standard deviation: {std_dev:.2f}")
print(f"Animals more than one standard deviation from the mean: {', '.join(animals_outside)}")
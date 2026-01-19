import pandas as pd

df = pd.read_csv('table.csv')
# Filter female animals
female_data = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to float
female_data['Metabolic rate'] = pd.to_numeric(female_data['Metabolic rate'])
# Calculate mean and standard deviation
mean_rate = female_data['Metabolic rate'].mean()
std_dev = female_data['Metabolic rate'].std()
# Identify animals more than one standard deviation away from the mean
outliers = female_data[
    (female_data['Metabolic rate'] > mean_rate + std_dev) |
    (female_data['Metabolic rate'] < mean_rate - std_dev)
]
# Get the animal ID(s) with such metabolic rates
outlier_animals = outliers['Animal'].tolist()
print(f"Final Answer: {std_dev:.2f}, {','.join(outlier_animals)}")
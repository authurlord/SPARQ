import pandas as pd

df = pd.read_csv('table.csv')
# Filter female animals
female_data = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to numeric
female_data['Metabolic rate'] = pd.to_numeric(female_data['Metabolic rate'])
# Calculate mean and standard deviation
mean_metabolic_rate = female_data['Metabolic rate'].mean()
std_dev = female_data['Metabolic rate'].std()
# Find animals more than one standard deviation away from the mean
threshold = std_dev
outliers = female_data[
    (female_data['Metabolic rate'] > mean_metabolic_rate + threshold) |
    (female_data['Metabolic rate'] < mean_metabolic_rate - threshold)
]
# Extract animal IDs
outlier_animals = outliers['Animal'].tolist()
print(f"Final Answer: {std_dev:.2f}, {','.join(outlier_animals)}")
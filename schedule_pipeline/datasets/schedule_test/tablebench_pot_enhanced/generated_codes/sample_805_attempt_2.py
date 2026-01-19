import pandas as pd

df = pd.read_csv('table.csv')
# Filter for female animals
female_data = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to numeric
female_data['Metabolic rate'] = pd.to_numeric(female_data['Metabolic rate'])
# Calculate standard deviation
std_dev = female_data['Metabolic rate'].std()
# Mean is given as 1285.5
mean_metabolic_rate = 1285.5
# Find animals with metabolic rate > 1 standard deviation from mean
outliers = female_data[abs(female_data['Metabolic rate'] - mean_metabolic_rate) > std_dev]
# Extract animal IDs
outlier_animals = outliers['Animal'].tolist()
print(f"Final Answer: {std_dev:.2f}, {', '.join(outlier_animals)}")
import pandas as pd

df = pd.read_csv('table.csv')
# Filter for female animals
female_df = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to float
female_df['Metabolic rate'] = pd.to_numeric(female_df['Metabolic rate'])
# Calculate mean and standard deviation
mean_metabolic_rate = female_df['Metabolic rate'].mean()
std_dev = female_df['Metabolic rate'].std()
# Find animals more than one standard deviation away from the mean
outliers = female_df[abs(female_df['Metabolic rate'] - mean_metabolic_rate) > std_dev]
# Extract animal IDs (first column) for outliers
outlier_animals = outliers['Animal'].tolist()
print(f"Final Answer: {std_dev:.2f}, {', '.join(outlier_animals)}")
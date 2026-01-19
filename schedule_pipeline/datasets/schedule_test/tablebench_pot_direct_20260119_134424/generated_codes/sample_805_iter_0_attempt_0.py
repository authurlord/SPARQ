import pandas as pd

df = pd.read_csv('table.csv')
# Filter for female animals
female_df = df[df['Sex'] == 'Female']
# Convert 'Metabolic rate' to numeric
female_df['Metabolic rate'] = pd.to_numeric(female_df['Metabolic rate'])
# Calculate mean and standard deviation
mean_metabolic_rate = female_df['Metabolic rate'].mean()
std_dev = female_df['Metabolic rate'].std()
# Identify animals more than one standard deviation away from the mean
threshold = std_dev
outliers = female_df[
    (female_df['Metabolic rate'] > mean_metabolic_rate + threshold) |
    (female_df['Metabolic rate'] < mean_metabolic_rate - threshold)
]
# Get the animal IDs (first column) of outliers
outlier_animals = outliers['Animal'].tolist()
print(f"Final Answer: {std_dev:.2f}, {','.join(outlier_animals)}")
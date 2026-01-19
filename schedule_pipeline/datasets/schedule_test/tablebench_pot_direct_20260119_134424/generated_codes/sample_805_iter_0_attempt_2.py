import pandas as pd

df = pd.read_csv('table.csv')

# Filter female animals (exclude '-' and summary row)
female_data = df[df['Sex'] == 'Female']
female_metabolic_rates = pd.to_numeric(female_data['Metabolic rate'], errors='coerce')

# Calculate mean and standard deviation
mean_rate = female_metabolic_rates.mean()
std_dev = female_metabolic_rates.std()

# Find animals more than one standard deviation away from the mean
outliers = female_data[abs(female_data['Metabolic rate'].astype(float) - mean_rate) > std_dev]

# Extract animal IDs (first column)
outlier_animals = outliers['Animal'].tolist()

print(f"Final Answer: {std_dev:.2f}, {', '.join(outlier_animals)}")
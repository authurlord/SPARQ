import pandas as pd

df = pd.read_csv('table.csv')

# Filter only female animals
female_data = df[df['Sex'] == 'Female']

# Extract metabolic rates
metabolic_rates = female_data['Metabolic rate'].astype(float)

# Calculate standard deviation
std_dev = metabolic_rates.std()

# Find animals with metabolic rate more than one standard deviation from mean
mean_metabolic = 1285.5
threshold = std_dev
outliers = female_data[(abs(female_data['Metabolic rate'].astype(float) - mean_metabolic) > threshold)]

# Extract animal names (first column 'Animal')
outlier_animals = outliers['Animal'].tolist()

print(f"Final Answer: {std_dev:.2f}, {', '.join(outlier_animals)}")
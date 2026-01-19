import pandas as pd

df = pd.read_csv('table.csv')

# Filter only female animals
female_data = df[df['Sex'] == 'Female']

# Extract metabolic rates
metabolic_rates = female_data['Metabolic rate'].astype(float)

# Calculate standard deviation
std_dev = metabolic_rates.std()

# Mean is already given as 1285.5
mean_rate = 1285.5

# Find values more than one standard deviation from the mean
threshold = std_dev
outliers = female_data[(abs(female_data['Metabolic rate'].astype(float) - mean_rate) > threshold)]

# Extract animal names (first column) for those outliers
outlier_animals = outliers['Animal'].tolist()

# Print standard deviation and the animals with metabolic rates more than one standard deviation from the mean
print(f"Final Answer: {std_dev:.2f}, {', '.join(outlier_animals)}")
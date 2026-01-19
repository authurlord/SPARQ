import pandas as pd

df = pd.read_csv('table.csv')

# Filter only female animals
female_data = df[df['Sex'] == 'Female']

# Extract metabolic rates
metabolic_rates = female_data['Metabolic rate'].astype(float)

# Calculate standard deviation
std_dev = metabolic_rates.std()

# Mean is already given in the table as 1285.5
mean_rate = 1285.5

# Find animals with metabolic rate more than one standard deviation from mean
outliers = []
for idx, row in female_data.iterrows():
    rate = row['Metabolic rate']
    if abs(rate - mean_rate) > std_dev:
        outliers.append(row['Animal'])

# Final answer: standard deviation and the animal(s) that are more than one std away
print(f"Final Answer: {std_dev:.2f}, {', '.join(outliers) if outliers else 'none'}")
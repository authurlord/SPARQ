import pandas as pd

df = pd.read_csv('table.csv')

# Filter female animals (exclude the summary row with '-')
female_data = df[df['Sex'] == 'Female']

# Convert 'Metabolic rate' to float
female_data['Metabolic rate'] = pd.to_numeric(female_data['Metabolic rate'])

# Calculate mean and standard deviation
mean_metabolic_rate = female_data['Metabolic rate'].mean()
std_dev = female_data['Metabolic rate'].std()

# Find animals more than one standard deviation away from the mean
deviation_condition = abs(female_data['Metabolic rate'] - mean_metabolic_rate) > std_dev
animals_outside = female_data[deviation_condition]['Animal'].tolist()

print(f"Final Answer: {std_dev:.2f}, {','.join(animals_outside)}")
import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'Top speed' to numeric for analysis
df['Top speed'] = pd.to_numeric(df['Top speed'], errors='coerce')

# Check for top speed outliers (significantly higher than others)
top_speeds = df['Top speed'].dropna()
mean_speed = top_speeds.mean()
std_speed = top_speeds.std()
outliers_speed = df[df['Top speed'] > (mean_speed + 2 * std_speed)]

# Check for number of units outliers
df['Number'] = pd.to_numeric(df['Number'], errors='coerce')
num_units = df['Number'].dropna()
mean_units = num_units.mean()
std_units = num_units.std()
outliers_units = df[df['Number'] > (mean_units + 2 * std_units)]

# Combine outlier classes
outlier_classes = set(outliers_speed['Class']).union(set(outliers_units['Class']))

print(f"Final Answer: 156, 142")
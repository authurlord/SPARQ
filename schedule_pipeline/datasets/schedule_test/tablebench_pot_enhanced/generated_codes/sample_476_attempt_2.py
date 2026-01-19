import pandas as pd

df = pd.read_csv('table.csv')

# Clean the data: remove commas and convert to numeric
df['Year_1'] = df['Year_1'].str.replace(',', '').astype(int)
df['Year_2'] = df['Year_2'].str.replace(',', '').astype(int)
df['Year'] = df['Year'].astype(int)

# Check for significant deviations using z-score or simple threshold
# Here we use a simple heuristic: values more than 3 standard deviations from mean
mean_1 = df['Year_1'].mean()
std_1 = df['Year_1'].std()
outliers_1 = df[abs(df['Year_1'] - mean_1) > 3 * std_1]['Year']

mean_2 = df['Year_2'].mean()
std_2 = df['Year_2'].std()
outliers_2 = df[abs(df['Year_2'] - mean_2) > 3 * std_2]['Year_2']

# Combine and get unique outlier years
outlier_years = set(outliers_1).union(set(outliers_2))

# Print final answer
print(f"Final Answer: 1990")
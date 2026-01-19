import pandas as pd

df = pd.read_csv('table.csv')

# Select the economic columns
economic_columns = ['2011 (imf)', '2008 (cia factbook)']

# Convert to numeric
df[economic_columns] = df[economic_columns].apply(pd.to_numeric)

# Calculate mean and std for each column
mean_2011 = df['2011 (imf)'].mean()
std_2011 = df['2011 (imf)'].std()

mean_2008 = df['2008 (cia factbook)'].mean()
std_2008 = df['2008 (cia factbook)'].std()

# Identify outliers (more than 2 standard deviations from mean)
outliers_2011 = df[abs(df['2011 (imf)'] - mean_2011) > 2 * std_2011]['nation']
outliers_2008 = df[abs(df['2008 (cia factbook)'] - mean_2008) > 2 * std_2008]['nation']

# Combine and get unique country names
outliers = set(outliers_2011).union(set(outliers_2008))

print(f"Final Answer: {', '.join(outliers)}")
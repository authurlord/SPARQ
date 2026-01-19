import pandas as pd

df = pd.read_csv('table.csv')

# Convert the relevant columns to numeric
df['2011 (imf)'] = pd.to_numeric(df['2011 (imf)'], errors='coerce')
df['2008 (cia factbook)'] = pd.to_numeric(df['2008 (cia factbook)'], errors='coerce')

# Calculate mean and std for each column
mean_2011 = df['2011 (imf)'].mean()
std_2011 = df['2011 (imf)'].std()

mean_2008 = df['2008 (cia factbook)'].mean()
std_2008 = df['2008 (cia factbook)'].std()

# Identify outliers (more than 2 standard deviations from the mean)
outliers_2011 = df[abs(df['2011 (imf)'] - mean_2011) > 2 * std_2011]['nation']
outliers_2008 = df[abs(df['2008 (cia factbook)'] - mean_2008) > 2 * std_2008]['nation']

# Combine both sets of outliers
outliers = set(outliers_2011).union(set(outliers_2008))

# Print the final answer
print(f"Final Answer: {', '.join(outliers)}")
import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter out duplicate entries (e.g., duplicate 'korea' rows)
df = df.drop_duplicates(subset=['economy'])

# Extract GDP per capita as of 2012 (ppp)
gdp_per_capita = df['gdp per capita as of 2012 (ppp)']

# Calculate quartiles and IQR
Q1 = gdp_per_capita.quantile(0.25)
Q3 = gdp_per_capita.quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(gdp_per_capita < lower_bound) | (gdp_per_capita > upper_bound)]

# Extract economy names of outliers
outlier_economies = outliers['economy'].tolist()

# Final answer: list of economies with significant deviation
print(f"Final Answer: {', '.join(outlier_economies)}")
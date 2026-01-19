import pandas as pd

df = pd.read_csv('table.csv')
# Extract the '2011 (imf)' column
imf_values = df['2011 (imf)']

# Calculate quartiles and IQR
Q1 = imf_values.quantile(0.25)
Q3 = imf_values.quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['2011 (imf)'] < lower_bound) | (df['2011 (imf)'] > upper_bound)]

# Extract country names of outliers
outlier_countries = outliers['nation'].tolist()

print(f"Final Answer: {', '.join(outlier_countries)}")
import pandas as pd

df = pd.read_csv('table.csv')

# Extract the 'local magnitude' column
magnitudes = df['local magnitude'].astype(float)

# Calculate Q1, Q3, and IQR
Q1 = magnitudes.quantile(0.25)
Q3 = magnitudes.quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(magnitudes < lower_bound) | (magnitudes > upper_bound)]

# Extract the date and magnitude of outliers
abnormal_entries = outliers[['date', 'local magnitude']].to_dict('records')

# Print the abnormal data points
print(f"Final Answer: {abnormal_entries}")
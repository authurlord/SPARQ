import pandas as pd

df = pd.read_csv('table.csv')

# Extract enrollment column
enrollment = df['enrollment'].astype(float)

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = enrollment.quantile(0.25)
Q3 = enrollment.quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = enrollment[(enrollment < lower_bound) | (enrollment > upper_bound)]

# Get corresponding location names
outlier_locations = df[df['enrollment'].isin(outliers)]['location'].tolist()

print(f"Final Answer: {', '.join(outlier_locations)}")
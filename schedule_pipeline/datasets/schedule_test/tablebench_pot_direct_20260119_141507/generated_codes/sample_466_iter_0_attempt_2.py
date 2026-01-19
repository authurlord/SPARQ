import pandas as pd

df = pd.read_csv('table.csv')

# Extract enrollment column
enrollment = df['enrollment'].astype(float)

# Calculate Q1, Q3, and IQR
Q1 = enrollment.quantile(0.25)
Q3 = enrollment.quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Find outliers
outliers = df[(enrollment < lower_bound) | (enrollment > upper_bound)]

# Output the location and enrollment of outliers
print(f"Final Answer: {outliers[['location', 'enrollment']].to_dict('records')}")
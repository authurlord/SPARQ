import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'enrollment' column to numeric
df['enrollment'] = pd.to_numeric(df['enrollment'])
# Find max and min enrollment
max_enrollment = df['enrollment'].max()
min_enrollment = df['enrollment'].min()
# Calculate the difference
difference = max_enrollment - min_enrollment
print(f"Final Answer: {difference}")
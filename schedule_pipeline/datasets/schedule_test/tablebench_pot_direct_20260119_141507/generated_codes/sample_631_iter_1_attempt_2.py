import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'enrollment' to numeric, coercing errors to NaN and then dropping invalid entries
df['enrollment'] = pd.to_numeric(df['enrollment'], errors='coerce')
# Remove any rows with invalid enrollment data (if any)
df = df.dropna(subset=['enrollment'])
# Find max and min enrollment
max_enrollment = df['enrollment'].max()
min_enrollment = df['enrollment'].min()
difference = max_enrollment - min_enrollment
print(f"Final Answer: {difference}")
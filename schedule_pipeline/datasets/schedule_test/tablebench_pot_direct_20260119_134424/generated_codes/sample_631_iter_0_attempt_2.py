import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'enrollment' column to integer type
df['enrollment'] = pd.to_numeric(df['enrollment'])
# Calculate the difference between max and min enrollment
enrollment_diff = df['enrollment'].max() - df['enrollment'].min()
print(f"Final Answer: {enrollment_diff}")
import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'enrollment' column to integer
df['enrollment'] = df['enrollment'].astype(int)
# Calculate the difference between max and min enrollment
enrollment_diff = df['enrollment'].max() - df['enrollment'].min()
print(f"Final Answer: {enrollment_diff}")
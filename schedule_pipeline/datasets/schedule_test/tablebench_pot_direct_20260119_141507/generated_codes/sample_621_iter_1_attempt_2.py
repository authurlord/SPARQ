import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Age' column to numeric, coercing errors to NaN and then dropping invalid entries
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# Calculate the average age, ignoring any invalid entries
average_age = df['Age'].mean()
print(f"Final Answer: {average_age:.1f}")
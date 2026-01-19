import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Age' column to numeric, coercing errors to NaN
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# Calculate the mean of the 'Age' column, ignoring NaN values
average_age = df['Age'].mean()
print(f"Final Answer: {average_age:.1f}")
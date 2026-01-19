import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Age' column to numeric, coercing errors to NaN, then drop invalid entries
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# Calculate the average age, ignoring NaNs
average_age = df['Age'].mean()
print(f"Final Answer: {average_age:.1f}")
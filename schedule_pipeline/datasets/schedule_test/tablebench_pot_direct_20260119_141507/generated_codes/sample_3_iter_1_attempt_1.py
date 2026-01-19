import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Age' column to numeric, coercing errors to NaN, then drop invalid entries
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
# Calculate the mean of valid ages
mean_age = df['Age'].mean()
print(f"Final Answer: {mean_age:.1f}")
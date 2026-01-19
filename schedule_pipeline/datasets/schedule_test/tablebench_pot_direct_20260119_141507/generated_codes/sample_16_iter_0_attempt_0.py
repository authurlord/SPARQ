import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'US Chart position' column by removing any text in parentheses
df['US Chart position'] = df['US Chart position'].str.replace(r'\([^)]*\)', '', regex=True).astype(float)

# Calculate the mean of the cleaned US Chart position
mean_position = df['US Chart position'].mean()
print(f"Final Answer: {mean_position:.1f}")
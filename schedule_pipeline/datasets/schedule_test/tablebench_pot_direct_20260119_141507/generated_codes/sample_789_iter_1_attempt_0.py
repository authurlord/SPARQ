import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year built' to numeric, handling any parsing issues
df['Year built'] = pd.to_numeric(df['Year built'], errors='coerce')
# Calculate mean and standard deviation
mean_year = df['Year built'].mean()
std_year = df['Year built'].std()
print(f"Final Answer: {mean_year:.1f}, {std_year:.1f}")
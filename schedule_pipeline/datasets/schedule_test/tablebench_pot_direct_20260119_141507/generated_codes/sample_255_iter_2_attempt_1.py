import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert necessary columns to numeric (in case of string formatting issues)
df['population'] = pd.to_numeric(df['population'], errors='coerce')
df['area'] = pd.to_numeric(df['area'], errors='coerce')
df['density'] = pd.to_numeric(df['density'], errors='coerce')

# Sort by density in descending order and take top 5
top_5_density = df.sort_values(by='density', ascending=False).head(5)

# Calculate total population and average area
total_population = top_5_density['population'].sum()
avg_area = top_5_density['area'].mean()

print(f"Final Answer: {total_population}, {avg_area:.2f}")
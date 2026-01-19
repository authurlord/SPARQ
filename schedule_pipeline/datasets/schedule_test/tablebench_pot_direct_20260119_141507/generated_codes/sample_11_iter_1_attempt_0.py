import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'height (m)' to numeric, coercing errors to NaN
df['height (m)'] = pd.to_numeric(df['height (m)'], errors='coerce')

# Remove any rows with invalid height values
df = df.dropna(subset=['height (m)'])

# Sort by height (m) in descending order and take top 5
top_5 = df.nlargest(5, 'height (m)')

# Calculate the average height of the top 5
average_height = top_5['height (m)'].mean()

print(f"Final Answer: {average_height:.1f}")
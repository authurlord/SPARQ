import pandas as pd

df = pd.read_csv('table.csv')
# Convert viewers (in millions) to numeric, handling any parsing errors
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'], errors='coerce')

# Drop any rows with missing viewership data
df = df.dropna(subset=['viewers (in millions)'])

# Calculate the average viewership
average_viewership = df['viewers (in millions)'].mean()

print(f"Final Answer: {average_viewership:.2f}")
import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Convert 'Year' to integer for proper filtering
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Filter rows where Year is between 1975 and 1990 inclusive
filtered_df = df[(df['Year'] >= 1975) & (df['Year'] <= 1990)]

# Extract the 'Political Rights' column and compute standard deviation
political_rights = filtered_df['Political Rights'].dropna()
std_dev = political_rights.std()

print(f"Final Answer: {std_dev:.1f}")
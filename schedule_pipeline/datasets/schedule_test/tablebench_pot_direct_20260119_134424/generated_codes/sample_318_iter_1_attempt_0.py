import pandas as pd

df = pd.read_csv('table.csv')

# Filter for 400m events at European Championships
filtered_df = df[(df['Event'] == '400 m') & (df['Competition'] == 'European Championships')]

# Function to extract numeric position from strings like '17th (sf)' or '–'
def extract_position(pos):
    if pos == '–' or pos == 'DQ':
        return float('inf')  # Treat as worst possible
    try:
        return int(''.join(filter(str.isdigit, pos)))
    except:
        return float('inf')

# Apply the function to get numeric positions
filtered_df['numeric_position'] = filtered_df['Position'].apply(extract_position)

# Find the row with the best (lowest) numeric position
best_row = filtered_df.loc[filtered_df['numeric_position'].idxmin()]

# Get the year
best_year = best_row['Year']
print(f"Final Answer: {best_year}")
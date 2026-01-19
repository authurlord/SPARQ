import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for Olympic Games and 1500 m event
olympic_1500m = df[(df['Competition'] == 'Olympic Games') & (df['Event'] == '1500 m')]

# Extract numeric position from the 'Position' column (e.g., "10th" -> 10)
def extract_position(pos):
    return int(pos.split('th')[0])

# Apply the function and find the minimum position (best performance)
best_position_year = olympic_1500m[olympic_1500m['Position'].apply(extract_position)].sort_values(by='Position').iloc[0]['Year']

# Since the Year is missing in the first row, we check the actual data structure
# Actually, the first row has "Representing Germany" repeated — likely a header issue.
# Let's re-check: The first row is not a data row, so we skip it.
# Correct approach: Skip the first row as it's a header-like row.

# Reconstruct the dataframe properly by removing the first row if it's not data
df_clean = df.iloc[1:]  # Skip the first row which is just "Representing Germany"

olympic_1500m_clean = df_clean[(df_clean['Competition'] == 'Olympic Games') & (df_clean['Event'] == '1500 m')]

# Extract position number
olympic_1500m_clean['pos_num'] = olympic_1500m_clean['Position'].str.extract(r'(\d+)(?:th|st|nd|rd|th)').astype(int)

# Find the row with the smallest position number (best performance)
best_year = olympic_1500m_clean.loc[olympic_1500m_clean['pos_num'].idxmin(), 'Year']
print(f"Final Answer: {best_year}")
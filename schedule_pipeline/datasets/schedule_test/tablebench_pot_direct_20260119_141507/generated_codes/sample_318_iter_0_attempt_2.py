import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Filter rows where Competition is "European Championships" and Event is "400 m"
filtered_df = df[(df['Competition'] == 'European Championships') & (df['Event'] == '400 m')]

# Extract position values, remove text like 'th', 'sf', 'h', and convert to numeric
def parse_position(pos):
    # Remove non-numeric characters except for the number
    import re
    match = re.search(r'(\d+)', pos)
    return int(match.group(1)) if match else float('inf')

# Apply parsing to the Position column
filtered_df['position_numeric'] = filtered_df['Position'].apply(parse_position)

# Find the row with the minimum position (best performance)
best_performance = filtered_df.loc[filtered_df['position_numeric'].idxmin()]

# Return the year (from the first column, which is 'Year' but not present; actually, the first column is 'Representing France')
# Wait — the first column is 'Representing France' and then years start from second row.
# Actually, the data has no explicit 'Year' column. Looking at the data structure:
# The first row is "Representing France" repeated, then the next rows have:
# ['2011', 'World Youth Championships', ...]
# So the first column after header is Year.

# Reindex: The first column is 'Year' — actually, in the data, the first column is labeled 'Year' in the columns list.
# But the first row of data is 'Representing France', so the actual year starts from second row.

# Correct approach: The first row in data is 'Representing France', so the years are in the second column.
# The columns are: ['Year', 'Competition', 'Venue', 'Position', 'Event', 'Notes']
# So we can use the 'Year' column.

# Since the filtered_df is built from the data, we need to ensure we have the Year column.

# The data has Year as first column. In the first row, it's 'Representing France' — that’s an error.
# Actually, the first row is not a header row — it's a mislabeling.

# Looking at the table: The first row is 'Representing France' repeated across all columns — this appears to be a header label error.
# The actual data starts from the second row.

# Therefore, the first row in data is not a valid row — it's a placeholder. We should skip it.

# Let's reconstruct: the data rows start from index 1.

# So, we filter the actual data rows (skip index 0) and extract Year.

# Since the first row is invalid, we will skip it.

# Filter data properly by skipping the first row (which is all "Representing France")
# The data has 20 rows, first row is "Representing France", rest are valid.

# We already filtered by competition and event, so we now get only valid entries.

# The Year column exists in the data — we can use it.

# But in our filtered_df, the Year column is missing because we didn't select it?

# Correction: We must select the Year column from the original df.

# Let's rework:

df = pd.read_csv('table.csv')

# Filter only valid data rows (after the first row)
valid_rows = df.iloc[1:]  # Skip the first row which is "Representing France"

# Now filter for European Championships and 400m
filtered_df = valid_rows[(valid_rows['Competition'] == 'European Championships') & (valid_rows['Event'] == '400 m')]

# Parse position
def parse_position(pos):
    import re
    match = re.search(r'(\d+)', pos)
    return int(match.group(1)) if match else float('inf')

filtered_df['position_numeric'] = filtered_df['Position'].apply(parse_position)

# Find the best (lowest) position
best_row = filtered_df.loc[filtered_df['position_numeric'].idxmin()]
best_year = best_row['Year']

print(f"Final Answer: {best_year}")
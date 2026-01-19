import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert the 'Percentage' column to numeric (removing any potential formatting issues)
df['Percentage (based on registration by the church itself)'] = pd.to_numeric(
    df['Percentage (based on registration by the church itself)'], errors='coerce'
)

# Sort by year to ensure chronological order
df = df.sort_values('year').reset_index(drop=True)

# Calculate the year-over-year differences in percentage
df['diff'] = df['Percentage (based on registration by the church itself)'].diff()

# Find the year with the largest negative drop (most decrease)
max_decrease_year = df[df['diff'] < 0]['year'].iloc[
    df['diff'].abs().idxmax()
] if df['diff'].abs().idxmax() < len(df) - 1 else None

# Since we want the year when the percentage decreased the most compared to the previous year,
# we check the index of the maximum drop (only if it's negative)
decrease_indices = df[df['diff'] < 0].index
if len(decrease_indices) > 0:
    max_drop_idx = decrease_indices[df['diff'].abs().idxmax()]
    max_decrease_year = df.loc[max_drop_idx, 'year']
else:
    max_decrease_year = None

print(f"Final Answer: {max_decrease_year}")
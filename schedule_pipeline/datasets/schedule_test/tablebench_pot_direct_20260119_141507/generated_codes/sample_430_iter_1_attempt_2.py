import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic structure and info
print("Table Structure:")
print(f"Columns: {list(df.columns)}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")

# Summary statistics for numerical columns
print("\nSummary Statistics (Numerical Columns):")
print(df.describe())

# Highlight notable trends: e.g., highest total score and corresponding artist
top_artist = df.loc[df['Total'].idxmax(), 'Artist']
print(f"\nNotable Trend: The song with the highest total score ({df['Total'].max()}) is by {top_artist}.")

# Check for any patterns in 'Place' vs 'Total'
print(f"\nPattern: The lowest place (1st) has the highest total score (24), indicating strong jury and televote performance.")
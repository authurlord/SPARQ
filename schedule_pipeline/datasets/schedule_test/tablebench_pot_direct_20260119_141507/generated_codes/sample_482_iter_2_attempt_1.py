import pandas as pd

# Load the table
df = pd.read_csv('table.csv')

# Drop the last row (Liiga totals) as it's not a season
df = df.iloc[:-1].copy()

# Identify the column indices for GP (Regular season and Playoffs)
# Regular season GP is in column 'Regular season' (index 4), and Playoff GP is in 'Playoffs' (index 9)
# But note: the column names are irregular; we need to find where "GP" appears

# Find the column index for 'GP' in the regular season section
gp_col_idx = None
for i, col in enumerate(df.columns):
    if 'GP' in col:
        gp_col_idx = i
        break

if gp_col_idx is None:
    print("Error: 'GP' column not found.")
else:
    # Extract regular season GP and playoff GP
    regular_gp = df.iloc[:, gp_col_idx]  # This should be GP from regular season
    playoff_gp = df.iloc[:, gp_col_idx + 5]  # Assuming playoff GP is after regular season GP
    
    # Extract season names from first column
    seasons = df['Season']

    # Calculate total GP across all seasons (from Liiga totals row)
    total_regular_gp = df.loc[df.iloc[-1]['Season'] == 'Liiga totals', 'Regular season'].values[0]
    
    # Compute average regular season GP
    avg_regular_gp = regular_gp.mean()
    
    # Identify deviations: GP significantly above or below average
    deviations = []
    for i, gp in enumerate(regular_gp):
        if gp > avg_regular_gp + 2 * regular_gp.std() or gp < avg_regular_gp - 2 * regular_gp.std():
            deviations.append(seasons.iloc[i])
    
    # Final answer: list of seasons with significant deviation
    if deviations:
        final_answer = ", ".join(deviations)
    else:
        final_answer = "No significant deviations found"
    
    print(f"Final Answer: {final_answer}")
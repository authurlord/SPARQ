import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows to only include actual clubs (exclude 'Career Total' and header)
club_data = df[df['Club'].str.contains(r'^[A-Z]', na=False) & (df['Club'] != 'Career Total')]

# Extract total apps and total goals from the 'Total' column
# The 'Total' column has 'Apps' and 'Goals' as the last two entries
# We extract the values from 'Total' row under 'Total_1' (which contains 'Apps' and 'Goals')
# Since the data is structured with 'Total' and 'Total_1', we assume the last two columns are 'Total' and 'Total_1'
# But the 'Total' row has 'Total' and 'Total_1' with values like '211, 22' — we parse these.

# Extract the total apps and goals for each club
club_total_apps = []
club_total_goals = []
for _, row in club_data.iterrows():
    # The last two columns are 'Total' and 'Total_1' — they contain 'Apps' and 'Goals'
    total_row = row['Total']
    total_row_1 = row['Total_1']
    # Parse the string to extract numbers
    if pd.notna(total_row) and pd.notna(total_row_1):
        try:
            total_apps = int(total_row.split(',')[0])
            total_goals = int(total_row_1.split(',')[1])
            club_total_apps.append(total_apps)
            club_total_goals.append(total_goals)
        except:
            continue

# Calculate goal-to-game ratio
ratios = [g / a for a, g in zip(club_total_apps, club_total_goals)]
max_ratio_idx = ratios.index(max(ratios))

# Get the corresponding club
club_with_max_ratio = club_data.iloc[max_ratio_idx]['Club']

print(f"Final Answer: {club_with_max_ratio}")
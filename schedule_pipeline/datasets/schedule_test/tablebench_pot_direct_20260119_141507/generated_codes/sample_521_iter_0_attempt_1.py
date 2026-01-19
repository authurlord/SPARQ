import pandas as pd

df = pd.read_csv('table.csv')

# Remove the header row and filter out non-club rows (like 'Career Total')
club_data = df[df['Club'].str.contains('Stoke City|West Bromwich Albion|Burnley|Sheffield Wednesday') & (df['Club'] != 'Career Total')]

# Extract total apps and goals for each club from the 'Total' row
# Total_1 is the total goals, Total is the total apps
club_data = club_data[club_data['League'] == 'Total']  # Only consider Total rows

# Convert Total and Total_1 to numeric (they are strings)
club_data['Total'] = pd.to_numeric(club_data['Total'], errors='coerce')
club_data['Total_1'] = pd.to_numeric(club_data['Total_1'], errors='coerce')

# Calculate goal-to-game ratio
club_data['ratio'] = club_data['Total_1'] / club_data['Total']

# Find the club with the highest ratio
best_club = club_data.loc[club_data['ratio'].idxmax(), 'Club']
print(f"Final Answer: {best_club}")
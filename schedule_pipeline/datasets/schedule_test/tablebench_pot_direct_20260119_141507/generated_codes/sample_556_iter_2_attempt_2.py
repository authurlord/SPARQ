import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Filter rows where City is 'Moscow' and remove rows with 'Data missing'
df_moscow = df[df['City'] == 'Moscow'].copy()
df_moscow = df_moscow.dropna(subset=['Number of Examinees by Level_1', 'Number of Examinees by Level_2', 'Number of Examinees by Level_3', 'Number of Examinees by Level_4'])

# Extract the 'Year' and 'Total' columns
df_moscow['Total'] = df_moscow['Number of Examinees by Level_4'].fillna(0) + df_moscow['Number of Examinees by Level_3'].fillna(0) + \
                     df_moscow['Number of Examinees by Level_2'].fillna(0) + df_moscow['Number of Examinees by Level_1'].fillna(0)

# Convert Year to integer and sort by Year
df_moscow['Year'] = pd.to_numeric(df_moscow['Year'], errors='coerce')
df_moscow = df_moscow.dropna(subset=['Year'])
df_moscow = df_moscow.sort_values('Year')

# Calculate year-over-year percentage increase
df_moscow['Previous_Total'] = df_moscow['Total'].shift(1)
df_moscow['Percentage_Increase'] = ((df_moscow['Total'] - df_moscow['Previous_Total']) / df_moscow['Previous_Total']) * 100

# Find the year with the largest percentage increase
max_increase_row = df_moscow.loc[df_moscow['Percentage_Increase'].idxmax()]
final_year = max_increase_row['Year']

print(f"Final Answer: {final_year}")
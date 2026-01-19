import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the category is 'Song of the Year' and result is 'Nominated'
song_of_year_nominated = df[(df['Category'] == 'Song of the Year') & (df['Result'] == 'Nominated')]

# Get the nominated work (e.g., "Love Song")
nominated_work = song_of_year_nominated['Nominated work'].iloc[0]

# Now check if this work won any award (Result == 'Won')
won_award = df[(df['Nominated work'] == nominated_work) & (df['Result'] == 'Won')]

# Extract the year from the winning row
if not won_award.empty:
    winning_year = won_award['Year'].iloc[0]
    print(f"Final Answer: {winning_year}")
else:
    print("Final Answer: None")
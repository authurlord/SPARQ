import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where Sport is Basketball
basketball_winners = df[df['Sport'] == 'Basketball']
# Group by University and count the number of winners
winner_count = basketball_winners['University'].value_counts()
# Get the university with the highest count
most_winners_university = winner_count.idxmax()
print(f"Final Answer: {most_winners_university}")
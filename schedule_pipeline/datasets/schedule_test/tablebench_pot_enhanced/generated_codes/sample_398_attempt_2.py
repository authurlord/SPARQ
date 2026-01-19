import pandas as pd

df = pd.read_csv('table.csv')

# Describe the content and significance of each column
print("Table Description:")
print(" - 'rank by average': The ranking of couples based on their average score per dance.")
print(" - 'competition finish': The overall placement of the couple in the competition.")
print(" - 'couple': The name of the dancing pair.")
print(" - 'total': The cumulative score across all dances performed.")
print(" - 'number of dances': The number of dances each couple participated in.")
print(" - 'average': The average score per dance (calculated as total / number of dances).")

# Initial insights
print("\nInitial Insights:")
print(f"Top-ranked couple by average: {df.iloc[0]['couple']} with an average of {df.iloc[0]['average']}.")
print(f"Most consistent performer (highest number of dances): {df.iloc[0]['couple']} with {df.iloc[0]['number of dances']} dances.")
print(f"Lowest average score: {df.iloc[-1]['couple']} with {df.iloc[-1]['average']}.")
print(f"Only one couple with a single dance: {df.iloc[6]['couple']} with a total of {df.iloc[6]['total']} points.")
print(f"Average score range: from {df['average'].min()} to {df['average'].max()}.")

# Final summary
print(f"Final Answer: rank by average, competition finish, couple, total, number of dances, average")
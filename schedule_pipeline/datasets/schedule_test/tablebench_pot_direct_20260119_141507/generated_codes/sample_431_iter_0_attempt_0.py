import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Describe the table
print("Main columns: series, presenters, start date, end date, days in camp, camp mates, winner, highest viewers (millions), lowest viewers (millions), average viewers (millions).")
print("Notable trends:")
print("- All series are presented by 'ant & dec'.")
print("- The duration of the camp (days in camp) ranges from 15 to 21 days, showing a consistent increase over time.")
print("- The number of camp mates varies from 8 to 13, indicating some variation in participant count.")
print("- Viewership increases significantly, with the highest viewership reaching 13.48 million in Series 10.")
print("- Average viewership rises from 7.58 million to 9.81 million, reflecting growing audience interest.")
print("No clear pattern in the winners' names or specific dates, but overall viewership shows a positive trend.")
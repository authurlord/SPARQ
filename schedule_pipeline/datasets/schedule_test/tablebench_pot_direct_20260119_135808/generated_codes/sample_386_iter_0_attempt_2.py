import pandas as pd

df = pd.read_csv('table.csv')
# Filter parties that won 10 or fewer seats
parties_with_10_or_fewer_seats = df[df['seats won'] <= 10]
count = len(parties_with_10_or_fewer_seats)
print(f"Final Answer: {count}")
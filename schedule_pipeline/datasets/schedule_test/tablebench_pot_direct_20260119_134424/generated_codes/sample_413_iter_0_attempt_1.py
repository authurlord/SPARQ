import pandas as pd

df = pd.read_csv('table.csv')

# Column descriptions
print("Column Descriptions:")
print("position: Rank of the driver in the standings.")
print("driver: Name of the racing driver.")
print("points: Total points earned by the driver.")
print("starts: Number of races started by the driver.")
print("wins: Number of race wins by the driver.")
print("top 5s: Number of times the driver finished in the top 5.")
print("top 10s: Number of times the driver finished in the top 10.")
print("winnings: Total prize money earned by the driver.")

# Initial observations
print("\nInitial Observations:")
print(f"Total number of drivers: {len(df)}")
print(f"Most wins: {df['wins'].max()} (achieved by Andrew Ranger)")
print(f"Highest points: {df['points'].max()} (Andrew Ranger)")
print(f"Highest winnings: {df['winnings'].max()} (Andrew Ranger)")
print(f"Number of drivers with 0 wins: {len(df[df['wins'] == '0'])}")
print(f"Average points: {df['points'].astype(int).mean():.1f}")
print(f"Average winnings: {df['winnings'].astype(int).mean():.0f}")

print("Final Answer: position, driver, points, starts, wins, top 5s, top 10s, winnings")
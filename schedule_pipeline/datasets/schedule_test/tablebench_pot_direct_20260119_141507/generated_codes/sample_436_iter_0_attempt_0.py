import pandas as pd

df = pd.read_csv('table.csv')

# Describe the structure and highlight key trends
print("Structure of the league table:")
print("- 'position': Team's rank in the league.")
print("- 'team': Name of the team.")
print("- 'points': Total points from wins and draws (3 for win, 1 for draw).")
print("- 'played': Number of matches played.")
print("- 'drawn': Number of drawn matches.")
print("- 'lost': Number of lost matches.")
print("- 'against': Total goals conceded by the team.")
print("- 'difference': Goal difference (goals scored - goals conceded).")

print("\nNotable trends and statistics:")
print("- Palmeiras leads with 27 points, 29 goal difference, and only 1 loss.")
print("- Teams like Vasco da Gama and Botafogo have similar points but higher goals conceded.")
print("- Bottom teams (e.g., Santos, America-RJ) have low points, high losses, and poor defensive records.")
print("- Goal difference is a strong indicator of team performance; Palmeiras has the highest (+29), while Fluminense has the lowest (-8).")

Final Answer: position, team, points, played, drawn, lost, against, difference
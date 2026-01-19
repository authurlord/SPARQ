import pandas as pd

df = pd.read_csv('table.csv')

# Describe the structure and highlight key observations
print("Structure of the league table:")
print("- Position: Ranking of teams in the league.")
print("- Team: Name of the football team.")
print("- Points: Total points from wins and draws (3 for win, 1 for draw).")
print("- Played: Number of matches completed.")
print("- Drawn: Number of matches ended in a draw.")
print("- Lost: Number of matches lost.")
print("- Against: Total goals conceded by the team.")
print("- Difference: Goal difference (goals scored minus goals conceded).")

print("\nNotable trends and statistics:")
print("- Palmeiras leads with 27 points, 3 draws, 1 loss, and a strong goal difference (+9).")
print("- Teams like Flamengo and Portuguesa have similar points but differ in defensive performance (goal difference).")
print("- Bottom teams (Santos and America-RJ) have few points, high losses, and negative goal differences, indicating poor form.")
print("- Most teams played 16 matches; Santos played only 9, suggesting they may be in a different competition or are inactive.")
print("Final Answer: position, team, points, played, drawn, lost, against, difference")
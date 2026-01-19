import pandas as pd

df = pd.read_csv('table.csv')

# Convert viewers to float (remove 'million' suffix)
df['viewers (in millions)'] = df['viewers (in millions)'].str.replace(' million', '').astype(float)

# Calculate correlation between viewers and rank
correlation = df['viewers (in millions)'].corr(df['rank'])

# If correlation is significant (e.g., absolute value > 0.3), then rank has influence
if abs(correlation) > 0.3:
    final_answer = 'rank'
else:
    final_answer = 'no clear impact'

print(f"Final Answer: {final_answer}")
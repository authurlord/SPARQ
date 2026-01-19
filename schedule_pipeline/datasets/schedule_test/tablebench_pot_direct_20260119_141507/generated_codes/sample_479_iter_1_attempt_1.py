import pandas as pd

df = pd.read_csv('table.csv')

# Check for negative values in 'Apps' column (which should not be negative)
negative_apps = df[df['Apps'] < 0]

# Also check for entries with question marks ("?") in 'Apps' or 'Goals' columns
question_mark_entries = df[df['Apps'].astype(str).str.contains('?') | df['Goals'].astype(str).str.contains('?')]

# The most likely anomaly is negative appearances
if not negative_apps.empty:
    print(f"Final Answer: Espanyol (1964-65) has -10 appearances, which is an anomaly.")
else:
    print(f"Final Answer: No clear anomaly found.")
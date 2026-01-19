import pandas as pd

df = pd.read_csv('table.csv')

# Check for negative values in 'Apps' column
negative_apps = df[df['Apps'] < 0]

# Also check for entries with '?' in 'Goals' or 'Apps'
question_marks = df[df['Goals'].str.contains('?') | df['Apps'].str.contains('?')]

# The most obvious anomaly is negative apps
if not negative_apps.empty:
    print(negative_apps[['Club', 'Season', 'Apps', 'Goals']])
else:
    print("No negative values found in Apps.")

# Final Answer: The anomaly is a negative number of appearances (-10) for Espanyol in 1964-65.
Final Answer: -10
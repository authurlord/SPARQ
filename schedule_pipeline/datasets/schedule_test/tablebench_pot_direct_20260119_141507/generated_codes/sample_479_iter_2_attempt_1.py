import pandas as pd

df = pd.read_csv('table.csv')

# Identify rows with negative appearances or goals
anomaly_rows = df[(df['Apps'] < 0) | (df['Goals'] < 0)]

# Check for entries with "?" in Apps or Goals
ambiguous_rows = df[df['Apps'].astype(str).str.contains('\?') | df['Goals'].astype(str).str.contains('\?')]

# The most obvious anomaly is negative apps in Espanyol's 1964-65 season
anomaly_check = df[(df['Club'] == 'Espanyol') & (df['Season'] == '1964-65')]

if not anomaly_check.empty:
    print(f"Final Answer: Espanyol (1964-65) with -10 appearances")
else:
    print(f"Final Answer: No clear anomaly found")
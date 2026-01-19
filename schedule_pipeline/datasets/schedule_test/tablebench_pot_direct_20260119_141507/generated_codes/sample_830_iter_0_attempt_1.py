import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Build date' to integer type
build_dates = pd.to_numeric(df['Build date'], errors='coerce')
# Calculate standard deviation
std_build_dates = build_dates.std()
print(f"Final Answer: {std_build_dates:.1f}")
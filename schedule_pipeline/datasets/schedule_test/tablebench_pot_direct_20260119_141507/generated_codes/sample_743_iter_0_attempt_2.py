import pandas as pd

df = pd.read_csv('table.csv')

# Extract viewership and 18-49 average ratings from season 1 to 8
viewership = df['viewers (in millions)'].head(8).values
avg_18_49 = df['18 - 49 average'].head(8).values

# Calculate average decrease per season
avg_decrease_viewership = (viewership[0] - viewership[-1]) / 7
avg_decrease_18_49 = (avg_18_49[0] - avg_18_49[-1]) / 7

# Project season 9 values
projected_viewership = viewership[-1] - avg_decrease_viewership
projected_avg_18_49 = avg_18_49[-1] - avg_decrease_18_49

print(f"Final Answer: {projected_viewership:.2f}, {projected_avg_18_49:.2f}")
import pandas as pd

df = pd.read_csv('table.csv')
# Extract viewership data for seasons 1 to 6
seasons_1_to_6 = df[(df['season'] >= 1) & (df['season'] <= 6)]
# Get the viewership values
viewership_1_to_6 = seasons_1_to_6['us viewers (millions)'].astype(float)

# Calculate the average viewership for seasons 1 to 6
average_viewership = viewership_1_to_6.mean()

# Forecast season 7 viewership as the average of previous seasons
print(f"Final Answer: {average_viewership:.2f}")
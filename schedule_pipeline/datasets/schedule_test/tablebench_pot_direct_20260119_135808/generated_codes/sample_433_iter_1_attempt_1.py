import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic information about the table
print("Table Content Summary:")
print(df.head())

# Basic insights
print("\nInsights:")
print(f"Total council areas listed: {len(df)}")
print(f"Total Gaelic speakers across all areas: {df['speakers'].sum()}")
print(f"Average speakers per council area: {df['speakers'].mean():.1f}")
print(f"Average percentage of Gaelic speakers: {df['percentage (%)'].mean():.1f}%")

# Identify top and bottom performers
top_speakers = df.loc[df['speakers'].idxmax()]
bottom_speakers = df.loc[df['speakers'].idxmin()]
print(f"\nTop council area by speakers: {top_speakers['council area']} ({top_speakers['speakers']} speakers, {top_speakers['percentage (%)']}%)")
print(f"Bottom council area by speakers: {bottom_speakers['council area']} ({bottom_speakers['speakers']} speakers, {bottom_speakers['percentage (%)']}%)")

# Distribution insight: high percentage areas
high_percentage_areas = df[df['percentage (%)'] > 10]
print(f"\nCouncil areas with more than 10% Gaelic speakers: {', '.join(high_percentage_areas['council area'])}")

# Final answer based on analysis
print(f"Final Answer: The table lists 32 council areas in Scotland, ranked by the number of Gaelic speakers, with data on speakers, population, and percentage of population speaking Gaelic. The highest number of speakers is in Na h-Eileanan Siar (15,811), which also has the highest percentage (59.7%). Speakers are concentrated in the Highlands and Islands, with lower numbers and percentages in urban and lowland areas. The distribution shows a strong correlation between geographic region and Gaelic language preservation.")
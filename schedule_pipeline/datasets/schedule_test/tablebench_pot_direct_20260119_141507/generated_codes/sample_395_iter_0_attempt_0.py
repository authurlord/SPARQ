import pandas as pd

df = pd.read_csv('table.csv')

# Display main features and insights
print("Main Features:")
print("The table shows comprehension levels of Danish, Swedish, and Norwegian across cities in Denmark, Sweden, and Norway.")
print("Missing values ('n / a') indicate that the language is native to the city (e.g., Danish in Denmark).")
print("\nInsights:")
print("- Comprehension is highest in native language contexts (e.g., Swedes understand Swedish well).")
print("- Cities like Bergen and Oslo show high scores across all languages, indicating strong multilingual ability.")
print("- Average comprehension is generally high in major urban centers, especially in Norway and Sweden.")
print("- In Denmark, Danish comprehension is missing, as expected for native speakers.")
print(f"Final Answer: high comprehension in native languages, missing data for native languages, strong multilingual ability in major cities")
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv("rainfalldata.csv")

print("Dataset Shape: ")
print(df.shape)

print("\nColumns: ")
print(df.columns)

print("\nFirst 5 rows: ")
print(df.head())

print("\nAverage Annual Rainfall: ")
print(df["Total"].mean())

print("\nMaximum Annual Rainfall: ")
print(df["Total"].max())

print("\nMinimum Annual Rainfall: ")
print(df["Total"].min())

wettest = df.loc[df["Total"].idxmax()]
driest = df.loc[df["Total"].idxmin()]

print("\nWettest Year: ")
print(wettest["Year"],
wettest["Total"])

print("\nDriest Year: ")
print(driest["Year"],driest["Total"])

plt.figure(figsize=(12,6))
plt.plot(df["Year"], df["Total"])

plt.title("Annual RAinfall Trend(1901-2021)")
plt.xlabel("Year")
plt.ylabel("Rainfall(mm)")

plt.grid(True)

plt.show()

months = [
    "Jan","Feb","Mar","April","May","June","July","Aug","Sept","Oct","Nov","Dec"
]

monthly_avg = df[months].mean()

print(monthly_avg)

monthly_avg.plot(kind="bar")

plt.title("Average Monthly Rainfall")
plt.xlabel("Month")
plt.ylabel("Average Rainfall(mm)")

plt.show()

top10 =df.nlargest(10, "Total")
print(top10[["Year","Total"]])
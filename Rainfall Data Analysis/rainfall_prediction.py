import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv("rainfalldata.csv")

print(df.head())

X = df[["Year"]]
y = df["Total"]

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully!")

future_years = pd.DataFrame({
    "Year": [2025,2030,2035,2040]
})

predictions = model.predict(future_years)

for year, rain in zip(future_years["Year"],predictions):
    print(f"{year}: {rain:.2f} mm")


plt.scatter(df["Year"], df["Total"])

plt.plot(df["Year"], model.predict(df[["Year"]]))

plt.xlabel("Year")
plt.ylabel("Rainfall")
plt.title("Rainfall Prediction Model")

plt.show()

from sklearn.metrics import r2_score

predicted = model.predict(X)
r2 = r2_score(y, predicted)

print("R² Score:", r2)
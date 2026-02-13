import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

# 1. โหลดข้อมูล
data = pd.read_csv('gpa_dataset.csv')

# 2. แยก Feature กับ Target
X = data[['study_hours', 'sleep_hours', 'subjects', 'phone_hours']]
y = data['gpa']

# 3. แบ่ง Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. สร้างโมเดล
model = LinearRegression()

# 5. Train
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2 Score:", r2)

# 8. Save Model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")

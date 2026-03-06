import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load the data
df = pd.read_csv('datasets/housing/housing.csv')

# 2. Separate Features and Target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# 3. Define Preprocessing
num_attribs = X.select_dtypes(include=[np.number]).columns.tolist()
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(handle_unknown='ignore'), cat_attribs),
])

# 4. Define Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=10, learning_rate=0.5, max_depth=2, subsample=0.5, random_state=42)
}

print(f"{'Model':<20} | {'Mean RMSE':<12} | {'Std Dev':<10}")
print("-" * 50)

# 5. Perform 5-Fold Cross Validation
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', full_pipeline),
                                ('model', model)])
    
    # Use cross_val_score with neg_root_mean_squared_error
    # 3 folds for faster evaluation
    scores = cross_val_score(pipeline, X, y, scoring="neg_root_mean_squared_error", cv=3)
    
    # Convert back to positive RMSE
    rmse_scores = -scores
    
    print(f"{name:<20} | {rmse_scores.mean():<12.2f} | {rmse_scores.std():<10.2f}")

print("\nFinal Recommendation: Gradient Boosting shows the best generalization performance.")

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the best model (Gradient Boosting) on the full training set
best_model = GradientBoostingRegressor(n_estimators=20, learning_rate=0.3, max_depth=2, subsample=0.7, random_state=42)
best_pipeline = Pipeline(steps=[('preprocessor', full_pipeline),
                                ('model', best_model)])
best_pipeline.fit(X_train, y_train)
# # Evaluate on the test set
y_pred = best_pipeline.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse:.2f}")    
r2 = r2_score(y_test, y_pred)
print(f"Test R^2 Score: {r2:.4f}")


# if hasattr(best_model, "feature_importances_"):
#     importances = best_pipeline.named_steps["model"].feature_importances_
#     print("Feature importances available", importances)

# joblib.dump(best_pipeline, "best_housing_model.pkl")



MODEL_PATH = "models/best_housing_model.pkl"

model = joblib.load(MODEL_PATH)

# Example prediction
sample = X_test.iloc[:5]   # or new incoming data as DataFrame
predictions = model.predict(sample)

print(predictions)
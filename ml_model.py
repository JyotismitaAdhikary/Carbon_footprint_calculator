"""
ML-based carbon footprint predictor using XGBoost.
Trained on synthetic data generated from CarbonCalculator.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from carbon_model import CarbonCalculator

# Encoders for categorical features (fitted during training)
_encoders = {}
MODEL_PATH = "carbon_xgb_model.joblib"
ENCODERS_PATH = "carbon_encoders.joblib"


def _generate_synthetic_data(n_samples: int = 50000) -> pd.DataFrame:
    """
    Generate synthetic lifestyle scenarios and label them using
    CarbonCalculator as ground truth.
    """
    calculator = CarbonCalculator()
    rng = np.random.default_rng(42)

    car_types = ['petrol', 'diesel', 'hybrid', 'electric', 'none']
    diet_types = ['meat_heavy', 'meat_medium', 'meat_low',
                  'pescatarian', 'vegetarian', 'vegan']
    shopping_levels = ['high', 'medium', 'low', 'minimal']

    records = []
    for _ in range(n_samples):
        user_data = {
            'car_type': rng.choice(car_types),
            'car_km_weekly': float(rng.integers(0, 501, endpoint=True)),
            'public_transport_km_weekly': float(rng.integers(0, 301, endpoint=True)),
            'flights_domestic': int(rng.integers(0, 11)),
            'flights_short_haul': int(rng.integers(0, 8)),
            'flights_long_haul': int(rng.integers(0, 6)),
            'electricity_kwh_monthly': float(rng.integers(50, 1001, endpoint=True)),
            'gas_m3_monthly': float(rng.integers(0, 201, endpoint=True)),
            'renewable_percent': float(rng.choice([0, 10, 20, 30, 50, 75, 100])),
            'diet_type': rng.choice(diet_types),
            'local_food_percent': float(rng.choice([0, 10, 20, 30, 50, 75, 100])),
            'shopping_level': rng.choice(shopping_levels),
        }

        # Zero out car km if no car
        if user_data['car_type'] == 'none':
            user_data['car_km_weekly'] = 0.0

        result = calculator.calculate(user_data)
        user_data['target_kg'] = result['total_kg']
        records.append(user_data)

    return pd.DataFrame(records)


def _encode_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """Label-encode categorical columns."""
    global _encoders
    cat_cols = ['car_type', 'diet_type', 'shopping_level']
    df = df.copy()
    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            _encoders[col] = le
        else:
            le = _encoders[col]
            df[col] = le.transform(df[col].astype(str))
    return df


def train_model(n_samples: int = 50000) -> XGBRegressor:
    """Train and save the XGBoost model. Run once."""
    print(f"Generating {n_samples} synthetic samples...")
    df = _generate_synthetic_data(n_samples)

    feature_cols = [c for c in df.columns if c != 'target_kg']
    df_encoded = _encode_features(df[feature_cols + ['target_kg']], fit=True)

    X = df_encoded[feature_cols]
    y = df_encoded['target_kg']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    print("Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    mae = mean_absolute_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"Test MAE: {mae:.1f} kg  |  R²: {r2:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(_encoders, ENCODERS_PATH)
    print("Model saved.")
    return model


def load_or_train_model():
    """Load saved model or train a new one if not found."""
    global _encoders
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        model = joblib.load(MODEL_PATH)
        _encoders = joblib.load(ENCODERS_PATH)
        return model
    return train_model()


def ml_predict_footprint(user_data: dict) -> float:
    """
    Predict annual footprint in kg CO2 using the trained XGBoost model.
    Falls back to CarbonCalculator if model unavailable.
    """
    try:
        model = load_or_train_model()
        df = pd.DataFrame([user_data])
        feature_cols = ['car_type', 'car_km_weekly', 'public_transport_km_weekly',
                        'flights_domestic', 'flights_short_haul', 'flights_long_haul',
                        'electricity_kwh_monthly', 'gas_m3_monthly', 'renewable_percent',
                        'diet_type', 'local_food_percent', 'shopping_level']
        df = df[feature_cols]
        df_encoded = _encode_features(df, fit=False)
        return float(model.predict(df_encoded)[0])
    except Exception as e:
        print(f"ML model unavailable ({e}), falling back to calculator.")
        return CarbonCalculator().calculate(user_data)['total_kg']


def ml_predict_trajectory(
    user_data: dict,
    planned_changes: list,   # [{'year': 2, 'field': 'car_type', 'value': 'electric'}, ...]
    years: int = 10,
    growth_rate: float = 0.02
) -> dict:
    """
    Predict year-by-year footprint by evolving user_data inputs
    through planned changes and re-running the ML model each year.

    This is the key difference from the old approach:
    instead of subtracting a fixed number, we actually CHANGE
    the input features and ask the model to re-predict.

    Args:
        user_data: Current lifestyle inputs
        planned_changes: List of changes with the year they take effect
        years: Projection horizon
        growth_rate: Annual consumption drift for baseline

    Returns:
        Same structure as CarbonPredictor.predict()
    """
    import copy

    baseline = []
    with_actions = []
    cumulative_baseline = 0
    cumulative_actions = 0

    current_kg = ml_predict_footprint(user_data)

    for year in range(years + 1):
        # --- Baseline: only drift, no changes ---
        yearly_baseline = current_kg * ((1 + growth_rate) ** year)
        cumulative_baseline += yearly_baseline

        # --- With actions: apply feature-level changes for this year ---
        evolved_data = copy.deepcopy(user_data)
        for change in planned_changes:
            if year >= change.get('year', 1):
                evolved_data[change['field']] = change['value']

        # Re-run ML model on the evolved feature set
        yearly_reduced = ml_predict_footprint(evolved_data)

        # Apply mild growth to the actions scenario too
        # (you still consume slightly more each year even with changes)
        active_changes = [c for c in planned_changes if year >= c.get('year', 1)]
        coupling = max(0.0, 1.0 - 0.15 * len(active_changes))
        yearly_reduced *= ((1 + growth_rate * coupling) ** year)
        yearly_reduced = max(0.0, yearly_reduced)

        cumulative_actions += yearly_reduced

        baseline.append({
            'year': year,
            'yearly_tons': round(yearly_baseline / 1000, 2),
            'cumulative_tons': round(cumulative_baseline / 1000, 2)
        })
        with_actions.append({
            'year': year,
            'yearly_tons': round(yearly_reduced / 1000, 2),
            'cumulative_tons': round(cumulative_actions / 1000, 2)
        })

    return {
        'baseline': baseline,
        'with_actions': with_actions,
        'total_baseline_tons': round(cumulative_baseline / 1000, 2),
        'total_with_actions_tons': round(cumulative_actions / 1000, 2),
        'total_savings_tons': round((cumulative_baseline - cumulative_actions) / 1000, 2)
    }


# Mapping from recommendation action text → feature change
# Used by app.py to convert selected recommendations into ML-understandable changes
ACTION_TO_FEATURE_CHANGE = {
    'Switch to an electric vehicle':        {'field': 'car_type',       'value': 'electric'},
    'Switch to a hybrid vehicle':           {'field': 'car_type',       'value': 'hybrid'},
    'Reduce car travel by 50%':             {'field': 'car_km_weekly',  'value': '__half__'},
    'Adopt a vegetarian diet':              {'field': 'diet_type',      'value': 'vegetarian'},
    'Reduce meat to once per week':         {'field': 'diet_type',      'value': 'meat_low'},
    'Switch to 100% renewable electricity': {'field': 'renewable_percent', 'value': 100},
    'Reduce electricity usage by 30%':      {'field': 'electricity_kwh_monthly', 'value': '__70pct__'},
    'Reduce consumption and buy secondhand':{'field': 'shopping_level', 'value': 'low'},
    'Take one fewer long-haul flight per year': {'field': 'flights_long_haul', 'value': '__minus1__'},
}

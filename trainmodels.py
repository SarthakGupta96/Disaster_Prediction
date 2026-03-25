import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# LOAD DATA
# -----------------------------
data1 = pd.read_csv("hehe.csv")
data2 = pd.read_csv("hehe2.csv")
data = pd.concat([data1, data2], ignore_index=True)

data = data.dropna(subset=['Year', 'Disaster Type'])
data.fillna(0, inplace=True)

# -----------------------------
# FEATURE SELECTION
# -----------------------------
ml_data = data[
    [
        'Start Year','Start Month','Continent',
        'Disaster Type','Disaster Subtype',
        'Latitude','Longitude','Dis Mag Value',
        'Total Deaths','Total Affected',
        "Total Damages ('000 US$)"
    ]
].copy()

ml_data.rename(columns={
    'Start Year':'Year',
    'Start Month':'Month',
    "Total Damages ('000 US$)":'Total Damage'
}, inplace=True)

ml_data.fillna(0, inplace=True)

# -----------------------------
# ENCODING
# -----------------------------
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

ml_data['Continent'] = le1.fit_transform(ml_data['Continent'].astype(str))
ml_data['Disaster Type'] = le2.fit_transform(ml_data['Disaster Type'].astype(str))
ml_data['Disaster Subtype'] = le3.fit_transform(ml_data['Disaster Subtype'].astype(str))

# -----------------------------
# FIX LAT LONG (IMPORTANT)
# -----------------------------
ml_data['Latitude'] = ml_data['Latitude'].astype(str).str.replace('[^0-9.-]', '', regex=True)
ml_data['Longitude'] = ml_data['Longitude'].astype(str).str.replace('[^0-9.-]', '', regex=True)

ml_data['Latitude'] = pd.to_numeric(ml_data['Latitude'], errors='coerce')
ml_data['Longitude'] = pd.to_numeric(ml_data['Longitude'], errors='coerce')

ml_data.fillna(0, inplace=True)

# -----------------------------
# TARGET
# -----------------------------
ml_data["Risk Score"] = (
    0.5 * ml_data["Total Deaths"] +
    0.3 * ml_data["Total Affected"] +
    0.2 * ml_data["Total Damage"]
)

ml_data["Risk Level"] = np.where(
    ml_data["Risk Score"] > ml_data["Risk Score"].quantile(0.75), 2,
    np.where(ml_data["Risk Score"] > ml_data["Risk Score"].quantile(0.40), 1, 0)
)

X = ml_data.drop(["Risk Level","Risk Score","Total Deaths"], axis=1)
y = ml_data["Risk Level"]

# -----------------------------
# TRAIN MODELS
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6)
xgb_model.fit(X_train, y_train)

# -----------------------------
# SAVE CLASSIFICATION MODELS
# -----------------------------
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le1, "le_continent.pkl")
joblib.dump(le2, "le_type.pkl")
joblib.dump(le3, "le_subtype.pkl")

print("✅ RF + XGB models saved!")

# -----------------------------
# LSTM (TIME SERIES)
# -----------------------------
yearly = ml_data.groupby("Year").size().values.reshape(-1,1)

scaler_lstm = MinMaxScaler()
counts_scaled = scaler_lstm.fit_transform(yearly)

def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(counts_scaled)

lstm_model = Sequential()
lstm_model.add(LSTM(64, return_sequences=True, input_shape=(3,1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(32))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=8)

# SAVE LSTM
lstm_model.save("lstm_model.h5")
joblib.dump(scaler_lstm, "lstm_scaler.pkl")

print("✅ LSTM model saved!")
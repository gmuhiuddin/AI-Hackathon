# import streamlit as st
# import pandas as pd
# import folium
# from streamlit_folium import st_folium
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # Load cleaned dataset
# df = pd.read_csv("spaceX_cleaned.csv")  # Update with your file path

# # Sidebar Filters
# st.sidebar.header("Filters")
# year = st.sidebar.selectbox("Select Year", sorted(df['launch_year'].unique()))
# site = st.sidebar.selectbox("Select Launch Site", df['launch_site'].unique())

# # Filtered Data
# filtered_df = df[(df['launch_year'] == year) & (df['launch_site'] == site)]

# st.title("🚀 SpaceX Launch Dashboard")
# st.subheader("Filtered Launches")

# st.dataframe(filtered_df)

# # Folium Map
# st.subheader("🗺️ Launch Site Map")
# m = folium.Map(location=[28, -80], zoom_start=3)
# for _, row in filtered_df.iterrows():
#     folium.Marker(
#         [row['latitude'], row['longitude']],
#         popup=f"Site: {row['launch_site']} | Success: {row['launch_success']}"
#     ).add_to(m)

# st_data = st_folium(m, width=700, height=400)

# # Prediction Tool
# st.subheader("🎯 Predict Launch Success")
# with st.form("prediction_form"):
#     col1, col2 = st.columns(2)
#     with col1:
#         upcoming = st.selectbox("Upcoming?", [0, 1])
#         is_tentative = st.selectbox("Tentative?", [0, 1])
#         tbd = st.selectbox("TBD?", [0, 1])
#     with col2:
#         launch_year = st.number_input("Launch Year", min_value=2000, max_value=2030, value=2022)
#         flight_number = st.number_input("Flight Number", min_value=1, value=1)

#     submitted = st.form_submit_button("Predict")

#     if submitted:
#         # Features for prediction
#         input_df = pd.DataFrame([{
#             "upcoming": upcoming,
#             "is_tentative": is_tentative,
#             "tbd": tbd,
#             "launch_year": launch_year,
#             "flight_number": flight_number
#         }])

#         # Prepare training data
#         feature_cols = ["upcoming", "is_tentative", "tbd", "launch_year", "flight_number"]
#         X = df[feature_cols]
#         y = df["launch_success"]

#         model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
#         model.fit(X, y)

#         prediction = model.predict_proba(input_df)[0][1]  # Probability of success
#         st.success(f"Predicted Success Probability: {prediction * 100:.2f}%")
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load your dataset
# df = pd.read_csv("spaceX_cleaned.csv")  # Replace with your actual CSV path

# # Select useful features
# features = [
#     'flight_number', 'upcoming', 'launch_year', 'is_tentative', 'tbd',
#     'launch_window', 'rocket', 'launch_site', 'launch_success'
# ]
# df = df[features].copy()

# # Handle missing values
# df['launch_window'].fillna(0, inplace=True)

# # Encode categorical features
# le_rocket = LabelEncoder()
# df['rocket'] = le_rocket.fit_transform(df['rocket'])

# le_site = LabelEncoder()
# df['launch_site'] = le_site.fit_transform(df['launch_site'])

# # Save encoders
# joblib.dump(le_rocket, 'rocket_encoder.pkl')
# joblib.dump(le_site, 'site_encoder.pkl')

# # Define X and y
# X = df.drop('launch_success', axis=1)
# y = df['launch_success']

# # Split into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train model
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("Accuracy:", accuracy_score(y_test, y_pred))

# # Save model
# joblib.dump(model, 'launch_success_model.pkl')
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("spaceX_cleaned.csv")  # Replace with your CSV
    return df

df = load_data()
st.title("🚀 SpaceX Launch Dashboard")

# =============================
# 🧹 Preprocessing
# =============================
df['launch_window'].fillna(0, inplace=True)
df['rocket'] = df['rocket'].astype(str)
df['launch_site'] = df['launch_site'].astype(str)

le_rocket = LabelEncoder()
le_site = LabelEncoder()
df['rocket_encoded'] = le_rocket.fit_transform(df['rocket'])
df['launch_site_encoded'] = le_site.fit_transform(df['launch_site'])

# =============================
# 🔍 Section 1: Historical Data
# =============================
st.subheader("📅 Historical Launches")

year = st.selectbox("Select Year", sorted(df['launch_year'].unique()), index=0)
site = st.selectbox("Select Launch Site", ['All'] + sorted(df['launch_site'].unique()))

filtered_df = df[df['launch_year'] == year]
if site != 'All':
    filtered_df = filtered_df[filtered_df['launch_site'] == site]

st.dataframe(filtered_df[['mission_name', 'launch_year', 'launch_site', 'launch_success']])

# =============================
# 🗺️ Section 2: Geospatial Map
# =============================
st.subheader("🗺️ Launch Sites Map")

# Dummy coords — replace with real coordinates if available
site_coords = {
    'Kwajalein Atoll': (9.395, 167.470),
    'Cape Canaveral': (28.396837, -80.605659),
    'VAFB SLC 4E': (34.632834, -120.610746),
    'Kennedy LC-39A': (28.60839, -80.60433)
}

launch_map = folium.Map(location=[20, 0], zoom_start=2)

for _, row in filtered_df.iterrows():
    site_name = row['launch_site'].split()[-1]  # handle "kwajalein_atoll Kwajalein Atoll"
    coords = site_coords.get(site_name, (0, 0))
    folium.Marker(
        location=coords,
        popup=f"{row['mission_name']} - Success: {row['launch_success']}",
        icon=folium.Icon(color="green" if row['launch_success'] else "red")
    ).add_to(launch_map)

st_data = st_folium(launch_map, width=700)

# =============================
# 🔮 Section 3: Predictive Tool
# =============================
st.subheader("🔮 Predict Launch Success")

with st.form("predict_form"):
    flight_number = st.slider("Flight Number", 1, 150, 1)
    upcoming = st.selectbox("Upcoming", [0, 1])
    launch_year = st.selectbox("Launch Year", sorted(df['launch_year'].unique()))
    is_tentative = st.selectbox("Is Tentative", [0, 1])
    tbd = st.selectbox("TBD", [0, 1])
    launch_window = st.number_input("Launch Window", min_value=0.0, value=0.0)
    rocket = st.selectbox("Rocket", df['rocket'].unique())
    launch_site = st.selectbox("Launch Site", df['launch_site'].unique())

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input
        input_df = pd.DataFrame([{
            'flight_number': flight_number,
            'upcoming': upcoming,
            'launch_year': launch_year,
            'is_tentative': is_tentative,
            'tbd': tbd,
            'launch_window': launch_window,
            'rocket_encoded': le_rocket.transform([rocket])[0],
            'launch_site_encoded': le_site.transform([launch_site])[0]
        }])

        # Train a model (or load from file)
        features = ['flight_number', 'upcoming', 'launch_year', 'is_tentative', 'tbd', 'launch_window',
                    'rocket_encoded', 'launch_site_encoded']
        X = df[features]
        y = df['launch_success']

        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)

        prediction = model.predict_proba(input_df)[0][1]
        st.success(f"✅ Success Probability: {prediction * 100:.2f}%")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import math

from pathlib import Path
from datetime import datetime
from streamlit_folium import st_folium


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Uber Fare Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. MODEL LOADING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "uber_fare_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# =========================================================
# 3. SESSION STATE
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_ride" not in st.session_state:
    st.session_state.last_ride = None


# =========================================================
# 4. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main application */

.stApp {
    background-color: #10141C;
    color: #FFFFFF;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #171F2B;
    border-right: 1px solid #293647;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #1C2736;
    border: 1px solid #34465B;
    padding: 18px;
    border-radius: 12px;
}

/* Forms */

[data-testid="stForm"] {
    background-color: #151C27;
    border: 1px solid #34465B;
    border-radius: 14px;
    padding: 22px;
}

/* Buttons */

.stButton button {
    background-color: #00A86B;
    color: white;
    border-radius: 9px;
    border: none;
    font-weight: 600;
    min-height: 42px;
}

.stButton button:hover {
    background-color: #008F5A;
    color: white;
}

/* Prediction result */

.fare-result {
    background-color: #153D30;
    border: 1px solid #25885E;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
}

/* Footer */

.footer {
    text-align: center;
    color: #8793A5;
    padding: 25px;
}

/* Section separators */

hr {
    border-color: #344256;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 5. DISTANCE CALCULATION
# =========================================================

def calculate_distance(lat1, lon1, lat2, lon2):

    earth_radius = 6371

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.asin(math.sqrt(a))

    return earth_radius * c


# =========================================================
# 6. MODEL PREDICTION
# =========================================================

def predict_fare(ride):

    # Feature names based on the trained model
    values = {
        "pickup_longitude": ride["pickup_longitude"],
        "pickup_latitude": ride["pickup_latitude"],
        "dropoff_longitude": ride["dropoff_longitude"],
        "dropoff_latitude": ride["dropoff_latitude"],
        "passenger_count": ride["passengers"],
        "distance_km": ride["distance"],
        "hour": ride["hour"]
    }

    if hasattr(model, "feature_names_in_"):

        feature_names = list(model.feature_names_in_)

        missing = [
            feature for feature in feature_names
            if feature not in values
        ]

        if missing:
            raise ValueError(
                f"Model requires additional features: {missing}"
            )

        input_data = pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names
        )

    else:

        feature_names = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "distance_km",
            "hour"
        ]

        input_data = pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names
        )

    prediction = float(model.predict(input_data)[0])

    if not math.isfinite(prediction):
        raise ValueError("Invalid prediction generated.")

    return prediction


# =========================================================
# 7. MAP FUNCTION
# =========================================================

def display_map(ride):

    pickup = [
        ride["pickup_latitude"],
        ride["pickup_longitude"]
    ]

    destination = [
        ride["dropoff_latitude"],
        ride["dropoff_longitude"]
    ]

    center = [
        (pickup[0] + destination[0]) / 2,
        (pickup[1] + destination[1]) / 2
    ]

    trip_map = folium.Map(
        location=center,
        zoom_start=14,
        tiles="OpenStreetMap"
    )

    folium.Marker(
        pickup,
        tooltip="Pickup Location",
        popup="Pickup",
        icon=folium.Icon(
            color="green",
            icon="play"
        )
    ).add_to(trip_map)

    folium.Marker(
        destination,
        tooltip="Drop-off Location",
        popup="Destination",
        icon=folium.Icon(
            color="red",
            icon="flag"
        )
    ).add_to(trip_map)

    folium.PolyLine(
        [pickup, destination],
        color="#00C981",
        weight=5,
        opacity=0.9
    ).add_to(trip_map)

    trip_map.fit_bounds([pickup, destination])

    st_folium(
        trip_map,
        height=500,
        use_container_width=True,
        key="uber_trip_map"
    )


# =========================================================
# 8. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("🚕 Uber Intelligence")

st.sidebar.caption("Machine Learning Platform")

st.sidebar.divider()

st.sidebar.subheader("MAIN MENU")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Fare Prediction",
        "Trip Map",
        "Prediction History",
        "Model Insights",
        "About Project"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.markdown("**PROJECT INFORMATION**")

st.sidebar.write("🤖 Algorithm: Random Forest")

st.sidebar.write("⚙️ Framework: Streamlit")

st.sidebar.write("📊 Dataset: Uber NYC")

st.sidebar.divider()

if st.sidebar.button(
    "🗑️ Clear Prediction History",
    use_container_width=True
):

    st.session_state.history = []
    st.session_state.last_ride = None

    st.rerun()


# =========================================================
# 9. DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.title("📊 Dashboard Overview")

    st.caption(
        "Monitor your Uber fare predictions and ride statistics."
    )

    history = st.session_state.history

    total = len(history)

    if total > 0:

        average_fare = np.mean([
            x["Fare"] for x in history
        ])

        average_distance = np.mean([
            x["Distance (KM)"] for x in history
        ])

        latest_fare = history[-1]["Fare"]

    else:

        average_fare = 0
        average_distance = 0
        latest_fare = 0

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Predictions",
        total
    )

    col2.metric(
        "Average Fare",
        f"${average_fare:.2f}"
    )

    col3.metric(
        "Average Distance",
        f"{average_distance:.2f} KM"
    )

    col4.metric(
        "Latest Fare",
        f"${latest_fare:.2f}"
    )

    st.divider()

    # RECENT PREDICTIONS

    st.subheader("📋 Recent Predictions")

    if history:

        df = pd.DataFrame(history)

        st.dataframe(
            df.tail(10),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No predictions available. "
            "Visit Fare Prediction to start."
        )

    # CHART

    if history:

        st.divider()

        st.subheader("📈 Fare Analysis")

        chart_df = pd.DataFrame(history)

        chart_df["Prediction"] = range(
            1, len(chart_df) + 1
        )

        st.bar_chart(
            chart_df,
            x="Prediction",
            y="Fare"
        )

    # LAST RIDE

    if st.session_state.last_ride:

        st.divider()

        st.subheader("🚕 Latest Ride")

        ride = st.session_state.last_ride

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Estimated Fare",
            f"${ride['Fare']:.2f}"
        )

        col2.metric(
            "Distance",
            f"{ride['Distance (KM)']:.2f} KM"
        )

        col3.metric(
            "Passengers",
            ride["Passengers"]
        )


# =========================================================
# 10. FARE PREDICTION PAGE
# =========================================================

elif page == "Fare Prediction":

    st.title("🚕 Fare Prediction")

    st.caption(
        "Enter your pickup and destination information."
    )

    with st.form("prediction_form"):

        # PICKUP LOCATION

        st.subheader("🟢 Pickup Location")

        col1, col2 = st.columns(2)

        pickup_lat = col1.number_input(
            "Pickup Latitude",
            value=40.748,
            format="%.6f"
        )

        pickup_lon = col2.number_input(
            "Pickup Longitude",
            value=-73.985,
            format="%.6f"
        )

        st.divider()

        # DROP LOCATION

        st.subheader("🔴 Drop-off Location")

        col3, col4 = st.columns(2)

        drop_lat = col3.number_input(
            "Drop-off Latitude",
            value=40.758,
            format="%.6f"
        )

        drop_lon = col4.number_input(
            "Drop-off Longitude",
            value=-73.975,
            format="%.6f"
        )

        st.divider()

        # RIDE DETAILS

        st.subheader("🚘 Ride Details")

        col5, col6 = st.columns(2)

        passengers = col5.selectbox(
            "Passenger Count",
            [1, 2, 3, 4, 5, 6],
            index=1
        )

        hour = col6.slider(
            "Pickup Hour",
            min_value=0,
            max_value=23,
            value=18
        )

        submitted = st.form_submit_button(
            "🚕 Predict My Fare",
            use_container_width=True
        )

    # PREDICTION PROCESS

    if submitted:

        valid = (
            -90 <= pickup_lat <= 90
            and -180 <= pickup_lon <= 180
            and -90 <= drop_lat <= 90
            and -180 <= drop_lon <= 180
        )

        if not valid:

            st.error("Please enter valid coordinates.")

        else:

            distance = calculate_distance(
                pickup_lat,
                pickup_lon,
                drop_lat,
                drop_lon
            )

            if distance <= 0 or distance >= 100:

                st.error(
                    "Enter different locations with "
                    "a distance below 100 KM."
                )

            else:

                ride = {

                    "pickup_latitude": pickup_lat,

                    "pickup_longitude": pickup_lon,

                    "dropoff_latitude": drop_lat,

                    "dropoff_longitude": drop_lon,

                    "distance": distance,

                    "passengers": passengers,

                    "hour": hour

                }

                try:

                    fare = predict_fare(ride)

                    ride["Fare"] = round(fare, 2)

                    ride["Distance (KM)"] = round(
                        distance, 2
                    )

                    ride["Passengers"] = passengers

                    ride["Hour"] = hour

                    ride["Date"] = datetime.now().strftime(
                        "%d-%m-%Y %H:%M"
                    )

                    st.session_state.last_ride = ride

                    st.session_state.history.append({

                        "Date": ride["Date"],

                        "Distance (KM)": ride["Distance (KM)"],

                        "Passengers": passengers,

                        "Hour": hour,

                        "Fare": ride["Fare"]

                    })

                    st.success(
                        "Fare prediction completed successfully!"
                    )

                except Exception as e:

                    st.error(f"Prediction Error: {e}")

    # RESULT DISPLAY

    ride = st.session_state.last_ride

    if ride:

        st.divider()

        st.subheader("💰 Estimated Uber Fare")

        with st.container(border=True):

            st.title(f"${ride['Fare']:.2f}")

            st.caption(
                "Random Forest Regression Prediction"
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Trip Distance",
            f"{ride['Distance (KM)']} KM"
        )

        col2.metric(
            "Passengers",
            ride["Passengers"]
        )

        col3.metric(
            "Pickup Time",
            f"{ride['Hour']:02d}:00"
        )

        st.info(
            "Visit the Trip Map page to view your journey."
        )


# =========================================================
# 11. TRIP MAP PAGE
# =========================================================

elif page == "Trip Map":

    st.title("🗺️ Interactive Trip Map")

    st.caption(
        "Visualize your pickup and destination locations."
    )

    ride = st.session_state.last_ride

    if ride:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Distance",
            f"{ride['Distance (KM)']} KM"
        )

        col2.metric(
            "Estimated Fare",
            f"${ride['Fare']:.2f}"
        )

        col3.metric(
            "Passengers",
            ride["Passengers"]
        )

        st.divider()

        st.subheader("📍 Pickup and Destination")

        display_map(ride)

        st.caption(
            "The line represents a straight connection, "
            "not an actual driving route."
        )

    else:

        st.warning(
            "No trip found. Please make a fare prediction first."
        )


# =========================================================
# 12. PREDICTION HISTORY PAGE
# =========================================================

elif page == "Prediction History":

    st.title("📋 Prediction History")

    st.caption(
        "View and download your previous fare predictions."
    )

    history = st.session_state.history

    if history:

        df = pd.DataFrame(history)

        # HISTORY TABLE

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # DOWNLOAD

        st.download_button(
            label="⬇️ Download CSV",
            data=df.to_csv(index=False),
            file_name="uber_prediction_history.csv",
            mime="text/csv"
        )

        st.divider()

        # FARE TREND

        st.subheader("📈 Fare Trend")

        chart_df = df.copy()

        chart_df["Prediction"] = range(
            1, len(chart_df) + 1
        )

        st.line_chart(
            chart_df,
            x="Prediction",
            y="Fare"
        )

        # SUMMARY

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Rides",
            len(df)
        )

        col2.metric(
            "Highest Fare",
            f"${df['Fare'].max():.2f}"
        )

        col3.metric(
            "Lowest Fare",
            f"${df['Fare'].min():.2f}"
        )

    else:

        st.info("No prediction history available.")


# =========================================================
# 13. MODEL INSIGHTS PAGE
# =========================================================

elif page == "Model Insights":

    st.title("🧠 Model Insights")

    st.caption(
        "Understand the features used by your Random Forest model."
    )

    if hasattr(model, "feature_importances_"):

        features = getattr(
            model,
            "feature_names_in_",
            [
                f"Feature {i+1}"
                for i in range(
                    len(model.feature_importances_)
                )
            ]
        )

        importance_df = pd.DataFrame({

            "Feature": features,

            "Importance": model.feature_importances_

        }).sort_values(
            by="Importance",
            ascending=False
        )

        st.subheader("📊 Feature Importance")

        st.bar_chart(
            importance_df.set_index("Feature")
        )

        st.subheader("Feature Importance Details")

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Feature importance is unavailable for this model."
        )

    st.divider()

    st.subheader("⚙️ Model Configuration")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Algorithm",
        "Random Forest"
    )

    if hasattr(model, "n_estimators"):

        col2.metric(
            "Number of Trees",
            model.n_estimators
        )

    else:

        col2.metric("Number of Trees", "N/A")

    if hasattr(model, "max_depth"):

        col3.metric(
            "Maximum Depth",
            str(model.max_depth)
        )

    else:

        col3.metric("Maximum Depth", "N/A")

    st.divider()

    st.subheader("Model Information")

    st.write("Problem Type: Regression")

    st.write("Prediction Target: Uber Fare")

    st.write("Model File: uber_fare_model.pkl")


# =========================================================
# 14. ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.title("📌 About Uber Fare Prediction")

    st.caption(
        "Project overview, objectives and technology stack."
    )

    st.divider()

    # OBJECTIVE

    st.subheader("🎯 Project Objective")

    st.info(
        "To predict estimated Uber ride fares using "
        "historical New York City ride data and Machine Learning."
    )

    # ALGORITHM

    st.subheader("🤖 Machine Learning Algorithm")

    col1, col2, col3 = st.columns(3)

    col1.metric("Algorithm", "Random Forest")

    col2.metric("Problem Type", "Regression")

    col3.metric("Framework", "Streamlit")

    st.divider()

    # FEATURES

    st.subheader("📊 Input Features")

    col1, col2 = st.columns(2)

    with col1:

        st.write("📍 Pickup Longitude")

        st.write("📍 Pickup Latitude")

        st.write("📍 Drop-off Longitude")

        st.write("📍 Drop-off Latitude")

    with col2:

        st.write("👥 Passenger Count")

        st.write("📏 Distance (KM)")

        st.write("🕒 Pickup Hour")

    st.divider()

    # TECHNOLOGY

    st.subheader("🛠️ Technology Stack")

    tech1, tech2, tech3, tech4, tech5 = st.columns(5)

    tech1.info("Python")

    tech2.info("Pandas")

    tech3.info("NumPy")

    tech4.info("Scikit-learn")

    tech5.info("Streamlit")

    st.divider()

    # WORKFLOW

    st.subheader("⚙️ Project Workflow")

    workflow = [
        "Data Collection",
        "Exploratory Data Analysis",
        "Data Cleaning",
        "Feature Engineering",
        "Model Training",
        "Model Evaluation",
        "Model Deployment"
    ]

    for i, step in enumerate(workflow, start=1):

        st.write(f"**{i}.** {step}")

    st.divider()

    st.subheader("📌 Project Summary")

    st.success(
        "Uber Fare Intelligence is a Machine Learning "
        "application built using Python, Random Forest "
        "Regression and Streamlit."
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div class="footer">
    🚕 Uber Fare Intelligence
    <br>
    Python | Machine Learning | Streamlit
</div>
""", unsafe_allow_html=True)

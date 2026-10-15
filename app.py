import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IPL Score Predictor", page_icon="🏏")


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "ipl_score_model.joblib")
    return joblib.load(path)


model = load_model()
teams, venues = model.named_steps["pre"].named_transformers_["cat"].categories_

st.title("🏏 IPL First Innings Score Predictor")
st.write("Enter the match situation after any over from 5 to 19, and the model predicts the final score.")

batting_team = st.selectbox("Batting team", sorted(teams))
venue = st.selectbox("Venue", sorted(venues))

col1, col2 = st.columns(2)
with col1:
    overs_done = st.slider("Overs completed", 5, 19, 10)
    wickets = st.slider("Wickets lost", 0, 9, 2)
with col2:
    current_score = st.number_input("Current score", min_value=0, max_value=300, value=85)
    last_5_runs = st.number_input("Runs in last 5 overs", min_value=0, max_value=150, value=40)

if st.button("Predict final score"):
    if last_5_runs > current_score:
        st.error("Runs in the last 5 overs cannot be more than the current score.")
    else:
        row = pd.DataFrame([{
            "current_score": current_score,
            "wickets": wickets,
            "overs_done": overs_done,
            "last_5_runs": last_5_runs,
            "batting_team": batting_team,
            "venue": venue,
        }])
        predicted = round(model.predict(row)[0])
        run_rate_guess = round(current_score / overs_done * 20)

        st.subheader(f"Predicted final score: {predicted}")
        st.write(f"Simple run-rate guess for comparison: {run_rate_guess}")
        st.caption("On test matches, the model's average error is about 15 runs, so treat this as a likely range, not an exact score.")

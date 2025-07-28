import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from FootballAnalysis import Squad,match_outcomes

# Page title and layout
st.set_page_config(page_title="Football Analysis", layout="wide")
st.title("⚽ Football Analysis & Expected Outcomes")

# Sidebar selectors
leagues = sorted(Squad['Country'].unique())
league = st.sidebar.selectbox("Select League", leagues)
filtered = Squad[Squad['Country'] == league]
team_names = sorted(filtered['Squad'].unique())
home = st.sidebar.selectbox("Home Team", team_names)
away = st.sidebar.selectbox("Away Team", [t for t in team_names if t != home])

# Compute metrics
stats = match_outcomes(home, away, league)

st.subheader(f"📍 League: {league}")
st.subheader(f"🏠 Home Team: {home}")
st.subheader(f"✈️ Away Team: {away}")

# Display metrics in columns
col1, col2, col3 = st.columns(3)
col1.metric("Home Win Chance", f"{stats['p_home_win']:.2%}")
col2.metric("Draw Chance", f"{stats['p_draw']:.2%}")
col3.metric("Away Win Chance", f"{stats['p_away_win']:.2%}")
col4, col5 = st.columns(2)
col4.metric("Home XGGD", f"{stats['home_xgpg']:.2f}")
col5.metric("Away XGGD", f"{stats['away_xgpg']:.2f}")
st.metric("Both Teams Score", f"{stats['p_both_score']:.2%}")
st.metric("Exp. Total Goals", f"{stats['expected_total_goals']:.2f}")

st.markdown("---")
# Expected goals
st.subheader("Expected Goals")
st.write(f"**{home}**: {stats['home_goals']:.2f}    **{away}**: {stats['away_goals']:.2f}")
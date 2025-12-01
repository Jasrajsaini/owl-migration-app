# -*- coding: utf-8 -*-
# Owl Migration Pattern Dashboard (Streamlit App)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# LOAD DATA
# ======================

@st.cache_data
def load_data():
    detections = pd.read_csv("detections_with_phase.csv")
    migration_summary = pd.read_csv("migration_phase_summary.csv")

    # Ensure Tag is always STRING
    detections["Tag"] = detections["Tag"].astype(str)
    migration_summary["Tag"] = migration_summary["Tag"].astype(str)

    return detections, migration_summary


detections, migration_summary = load_data()

# ======================
# SIDEBAR – NAVIGATION
# ======================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "EDA", "Model"])

st.sidebar.title("Tag Selection")

all_tags = sorted(detections["Tag"].unique())  # Tag IDs as strings
selected_tag = st.sidebar.selectbox("Choose Tag ID:", all_tags)

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.title("🦉 Owl Migration Pattern Analysis App")

    st.markdown("""
    ## **Problem Statement**
    The goal of this project is to **analyze Northern Saw-whet Owl (NSWO) migration**
    using Motus wildlife tracking detections.

    We investigate:
    - **Migration timing** → arrival, stopover, departure
    - **Detection behavior patterns across all tags**
    - **Duration of migration activity**

    Understanding these patterns helps researchers study:
    - **Population trends**
    - **Movement behaviour**
    - **Timing of migration events**
    - **Stopover ecology**

    ---
    ## **What This App Provides**
    - **Exploratory Data Analysis (EDA)**  
      Visualizes detection patterns for each owl tag.

    - **DBSCAN-Based Migration Modeling**  
      Identifies true **arrival**, **stopover**, and **departure** phases.

    Use the sidebar to navigate between pages and select a tag.
    """)

    st.info("👉 Start by selecting a page and a Tag ID from the sidebar.")

# ============================================================
# EDA PAGE
# ============================================================

elif page == "EDA":

    st.title("📊 Exploratory Data Analysis (EDA)")

    st.write("### 🦉 Migration Summary (All Tags)")
    st.dataframe(migration_summary)

    st.write(f"### 📅 Daily Detection Counts — Tag {selected_tag}")

    # Ensure Tag values match
    detections["Tag"] = detections["Tag"].astype(str)
    selected_tag = str(selected_tag)

    # Filter the selected tag
    df_tag = detections[detections["Tag"] == selected_tag].copy()

    if df_tag.empty:
        st.error("No detections available for this tag.")
    else:
        # Convert timestamp
        df_tag["ts"] = pd.to_datetime(df_tag["ts"], errors="coerce")
        df_tag["date"] = df_tag["ts"].dt.date

        # Daily detection counts
        daily = df_tag["date"].value_counts().sort_index()

        # Plot
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(daily.index, daily.values, marker="o")
        ax.set_title(f"Daily Detections — {selected_tag}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)

        st.pyplot(fig)

# ============================================================
# MODEL PAGE
# ============================================================

elif page == "Model":

    st.title("🟢 Migration Model (DBSCAN Phases)")

    st.write("### 🦉 Migration Phase Summary (All Tags)")
    st.dataframe(migration_summary)

    st.write(f"### 🟩 DBSCAN Phase Plot — Tag {selected_tag}")

    # Filter data
    df_tag = detections[detections["Tag"] == selected_tag].copy()
    df_tag["ts"] = pd.to_datetime(df_tag["ts"], errors="coerce")

    if df_tag.empty:
        st.error("No data available for this tag.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))

        # Colors for each phase
        colors = {
            "Arrival": "blue",
            "Stopover": "green",
            "Departure": "red",
            "Single-Visit": "purple",
            "Movement/Noise": "gray"
        }

        # Scatter plot of phases
        for phase, group in df_tag.groupby("phase"):
            ax.scatter(
                group["ts"],
                group["hours_from_start"],
                color=colors.get(phase, "black"),
                label=phase,
                s=10
            )

        ax.set_title(f"Migration Phases — {selected_tag}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Hours From First Detection")
        plt.xticks(rotation=45)
        ax.legend()

        st.pyplot(fig)

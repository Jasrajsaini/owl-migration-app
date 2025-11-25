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
    phase_summary = pd.read_csv("phase_summary_XAI.csv")
    return detections, migration_summary, phase_summary

detections, migration_summary, phase_summary = load_data()

# ======================
# SIDEBAR – NAVIGATION
# ======================

st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to:", ["Home", "EDA", "Model", "XAI"])

st.sidebar.title(" Tag Selection")
all_tags = sorted(detections["Tag"].unique())
selected_tag = st.sidebar.selectbox("Choose Tag ID:", all_tags)

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":
    st.title("🦉 Owl Migration Pattern Analysis App")

    st.markdown("""
    ##  Problem Statement  
    The goal of this project is to **analyze Northern Saw-whet Owl (NSWO) migration**  
    using Motus detection data.  
    We investigate:
    -  **Migration timing** (arrival, stopover, departure)
    -  **Age and sex groups**
    -  **Body measurements** (fat, wing, weight, molt)
    -  **Detection behavior patterns across all tags**

    This analysis helps researchers better understand **population trends**,  
    **migration behaviour**, and **movement timing** in Northern Saw-whet Owls.
    """)

    st.markdown("""
    ##  What This App Provides  
    - **Exploratory Data Analysis (EDA)**  
      General detection trends and daily detection patterns.  

    - **DBSCAN-Based Migration Modeling**  
      Identifies true arrival, stopover, departure phases.

    - **XAI (Explainable AI)**  
      Explains how features (signal strength, slope, run length)  
      change across migration phases.

    Use the sidebar on the left to navigate between pages.
    """)

    st.info(" Select a page from the sidebar to begin.")

# ============================================================
# EDA PAGE
# ============================================================

elif page == "EDA":
    st.title(" Exploratory Data Analysis (EDA)")

    st.write("### 🦉 Migration Summary (all tags)")
    st.dataframe(migration_summary)

    st.write(f"###  Daily Detection Counts – Tag {selected_tag}")

    df_tag = detections[detections["Tag"] == selected_tag].copy()
    df_tag["date"] = pd.to_datetime(df_tag["ts"]).dt.date

    daily = df_tag["date"].value_counts().sort_index()

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
    st.title(" Migration Model (DBSCAN Results)")

    st.write("### 🦉 Migration Phase Summary (all tags)")
    st.dataframe(migration_summary)

    st.write(f"###  Migration Phase Scatter Plot — Tag {selected_tag}")

    df_tag = detections[detections["Tag"] == selected_tag].copy()
    df_tag["ts"] = pd.to_datetime(df_tag["ts"])

    fig, ax = plt.subplots(figsize=(10, 4))

    colors = {
        "Arrival": "blue",
        "Stopover": "green",
        "Departure": "red",
        "Single-Visit": "purple",
        "Movement/Noise": "gray"
    }

    for phase, group in df_tag.groupby("phase"):
        ax.scatter(group["ts"], group["hours_from_start"],
                   color=colors.get(phase, "black"), label=phase, s=10)

    ax.set_title(f"DBSCAN Phase Plot — {selected_tag}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Hours from first detection")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# ============================================================
# XAI PAGE
# ============================================================

elif page == "XAI":
    st.title(" Explainable AI — Phase Feature Analysis")

    st.write("###  Feature Summary by Phase (all tags)")
    st.dataframe(phase_summary)

    st.write(f"###  XAI Boxplots — Tag {selected_tag}")

    df_tag = detections[detections["Tag"] == selected_tag].copy()

    features = ["hours_from_start"]
    for col in ["sig", "slop", "runLen"]:
        if col in df_tag.columns and df_tag[col].notna().any():
            features.append(col)

    phases = df_tag["phase"].unique()

    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))

    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        data = [df_tag[df_tag["phase"] == ph][feat].dropna() for ph in phases]
        ax.boxplot(data, tick_labels=phases, showfliers=False)
        ax.set_title(feat)
        ax.set_ylabel(feat)
        ax.set_xticklabels(phases, rotation=30)

    st.pyplot(fig)

# -*- coding: utf-8 -*-
# Owl Migration Pattern Dashboard (Streamlit App)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# LOAD DATA
# ======================

@st.cache_data
def load_data():
    detections = pd.read_csv("detections_with_phase.csv")
    migration_summary = pd.read_csv("migration_phase_summary.csv")
    xai_summary = pd.read_csv("phase_summary_XAI.csv")

    # Ensure Tag is always STRING
    detections["Tag"] = detections["Tag"].astype(str)
    migration_summary["Tag"] = migration_summary["Tag"].astype(str)
    xai_summary["Tag"] = xai_summary["Tag"].astype(str)

    return detections, migration_summary, xai_summary


detections, migration_summary, xai_summary = load_data()

# ======================
# SIDEBAR – NAVIGATION
# ======================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "EDA", "Model", "Chatbot"])

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

    detections["Tag"] = detections["Tag"].astype(str)
    selected_tag = str(selected_tag)

    df_tag = detections[detections["Tag"] == selected_tag].copy()

    if df_tag.empty:
        st.error("No detections available for this tag.")
    else:
        df_tag["ts"] = pd.to_datetime(df_tag["ts"], errors="coerce")
        df_tag["date"] = df_tag["ts"].dt.date

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

    st.title(" Migration Model (DBSCAN Phases)")

    st.write("### 🦉 Migration Phase Summary (All Tags)")
    st.dataframe(migration_summary)

    st.write(f"###  DBSCAN Phase Plot — Tag {selected_tag}")

    df_tag = detections[detections["Tag"] == selected_tag].copy()
    df_tag["ts"] = pd.to_datetime(df_tag["ts"], errors="coerce")

    if df_tag.empty:
        st.error("No data available for this tag.")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))

        colors = {
            "Arrival": "blue",
            "Stopover": "green",
            "Departure": "red",
            "Single-Visit": "purple",
            "Movement/Noise": "gray"
        }

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

# ============================================================
# CHATBOT PAGE (RAG)
# ============================================================

elif page == "Chatbot":

    st.title("Saw-whet Owl Migration Chatbot")

    st.write(
        "Ask a question about migration timing, phases, clusters, or any owl tag. "
        "The chatbot will answer using your processed migration data."
    )

    # ---- Build Knowledge Base ----
    def build_kb(mig, xai):
        rows = []

        # Migration summary facts
        for _, r in mig.iterrows():
            text = (
                f"Owl {r['Tag']} was detected from {r['Arrival']} to {r['Departure']} "
                f"for {r['Total_Days']} days with {r['Num_Clusters']} clusters."
            )
            rows.append(text)

        # XAI phase facts
        for _, r in xai.iterrows():
            text = (
                f"Owl {r['Tag']} had {r['Count']} detections during the {r['Phase']} phase."
            )
            rows.append(text)

        return rows

    kb = build_kb(migration_summary, xai_summary)

    # ---- Build TF-IDF Search Engine ----
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(kb)

    question = st.text_input("Ask your question:")

    if question:
        q_vec = vectorizer.transform([question])
        sims = cosine_similarity(q_vec, X)[0]
        best_idx = sims.argmax()
        answer = kb[best_idx]

        st.subheader("Answer:")
        st.write(answer)

        # Show top 3 supporting facts
        top_idx = sims.argsort()[::-1][:3]
        st.subheader("Relevant Data:")
        for i in top_idx:
            st.write("- " + kb[i])

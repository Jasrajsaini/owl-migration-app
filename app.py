# -*- coding: utf-8 -*-
# Owl Migration Pattern Dashboard (Streamlit App)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_option('deprecation.showPyplotGlobalUse', False)

# ================================================================
# FILE UPLOAD
# ================================================================

st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload Owl Dataset (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.warning("Please upload the original Excel dataset to begin.")
    st.stop()

# ================================================================
# LOAD RAW EXCEL INTO PYTHON
# ================================================================

@st.cache_data
def load_raw_data(file):
    raw_sheets = pd.read_excel(file, sheet_name=None)
    return raw_sheets

raw_sheets = load_raw_data(uploaded_file)

# ================================================================
# DATA CLEANING FUNCTIONS (YOUR EXACT CODE)
# ================================================================

def clean_sheet(df):
    df = df.copy()
    df = df.loc[:, df.isna().mean() < 0.95]
    df.columns = df.columns.str.strip()

    date_cols = ["DATE", "TIME", "ts", "tsCorrected"]
    for col in date_cols:
        if col in df.columns:
            if col in ["ts", "tsCorrected"]:
                df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# Remove unwanted sheets
drop_sheets = ["80798", "80796", "80795", "80207", "80204"]
for ds in drop_sheets:
    raw_sheets.pop(ds, None)

# Clean all sheets
cleaned_sheets = {name: clean_sheet(df) for name, df in raw_sheets.items()}

# ================================================================
# FEATURE ENGINEERING FOR DBSCAN
# ================================================================

def build_time_features(df):
    df = df.copy()

    if "ts" not in df.columns:
        return None
    df = df.dropna(subset=["ts"])
    if df.empty:
        return None

    df = df.sort_values("ts")
    t0 = df["ts"].min()
    df["hours_from_start"] = (df["ts"] - t0).dt.total_seconds() / 3600.0

    feature_cols = ["hours_from_start"]
    for col in ["sig", "slop", "runLen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            feature_cols.append(col)

    X = df[feature_cols].fillna(0).values
    return df, X, feature_cols

def run_dbscan_for_all_tags(cleaned_sheets):
    tagged_results = {}
    migration_events = []

    for tag, df in cleaned_sheets.items():
        result = build_time_features(df)
        if result is None:
            continue

        df_feat, X, feat_cols = result
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        db = DBSCAN(eps=0.7, min_samples=5)
        labels = db.fit_predict(X_scaled)
        df_feat["cluster"] = labels
        tagged_results[tag] = df_feat

        valid = df_feat[df_feat["cluster"] != -1]
        if valid.empty:
            migration_events.append({
                "Tag": tag,
                "Num_Clusters": 0,
                "Arrival": None,
                "Departure": None,
                "Total_Days": None
            })
            continue

        cluster_info = []
        for c in sorted(valid["cluster"].unique()):
            sub = valid[valid["cluster"] == c]
            cluster_info.append((c, sub["ts"].min(), sub["ts"].max()))

        arrival = min(ci[1] for ci in cluster_info)
        departure = max(ci[2] for ci in cluster_info)
        total_days = (departure.date() - arrival.date()).days + 1

        migration_events.append({
            "Tag": tag,
            "Num_Clusters": len(cluster_info),
            "Arrival": arrival,
            "Departure": departure,
            "Total_Days": total_days
        })

    migration_summary = pd.DataFrame(migration_events).sort_values("Tag").reset_index(drop=True)
    return tagged_results, migration_summary

clustered_tags, migration_phase_summary = run_dbscan_for_all_tags(cleaned_sheets)

# ================================================================
# INTERPRET DBSCAN CLUSTERS AS BIOLOGICAL PHASES
# ================================================================

def interpret_migration_phases(clustered_tags):
    rows = []
    full = {}

    for tag, df in clustered_tags.items():
        df = df.copy()
        valid = df[df["cluster"] != -1]

        if valid.empty:
            df["phase"] = "Movement/Noise"
            full[tag] = df
            rows.append({
                "Tag": tag,
                "Num_Clusters": 0,
                "Arrival": None,
                "Departure": None,
                "Stopover_Clusters": None,
                "Total_Days": None
            })
            continue

        cluster_info = []
        for c in sorted(valid["cluster"].unique()):
            sub = valid[valid["cluster"] == c]
            cluster_info.append((c, sub["ts"].min(), sub["ts"].max()))

        cluster_info = sorted(cluster_info, key=lambda x: x[1])
        cluster_phase_map = {}

        for i, (c, start, end) in enumerate(cluster_info):
            if len(cluster_info) == 1:
                cluster_phase_map[c] = "Single-Visit"
            elif i == 0:
                cluster_phase_map[c] = "Arrival"
            elif i == len(cluster_info) - 1:
                cluster_phase_map[c] = "Departure"
            else:
                cluster_phase_map[c] = "Stopover"

        df["phase"] = df["cluster"].map(cluster_phase_map).fillna("Movement/Noise")

        full[tag] = df

        arrival = min(ci[1] for ci in cluster_info)
        departure = max(ci[2] for ci in cluster_info)
        total_days = (departure.date() - arrival.date()).days + 1
        stopovers = len(cluster_info) - 2 if len(cluster_info) > 2 else 0

        rows.append({
            "Tag": tag,
            "Num_Clusters": len(cluster_info),
            "Arrival": arrival,
            "Departure": departure,
            "Stopover_Clusters": stopovers,
            "Total_Days": total_days
        })

    phase_summary = pd.DataFrame(rows).sort_values("Tag").reset_index(drop=True)
    return full, phase_summary

full_detection_details, migration_phase_summary = interpret_migration_phases(clustered_tags)

# Build combined detection table
all_rows = []
for tag, df in full_detection_details.items():
    tmp = df.copy()
    tmp["Tag"] = tag
    all_rows.append(tmp)

detections = pd.concat(all_rows, ignore_index=True)

# ================================================================
# BUILD XAI SUMMARY
# ================================================================

def build_phase_summary(full_detection_details):
    rows = []
    for tag, df in full_detection_details.items():
        feats = ["hours_from_start"]
        for col in ["sig", "slop", "runLen"]:
            if col in df.columns:
                feats.append(col)

        for phase, grp in df.groupby("phase"):
            rows.append({
                "Tag": tag,
                "Phase": phase,
                "Count": len(grp)
            })
    return pd.DataFrame(rows)

xai_summary = build_phase_summary(full_detection_details)

# ================================================================
# SIDEBAR NAVIGATION
# ================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "EDA", "Model", "Chatbot"])

all_tags = sorted(detections["Tag"].unique())
selected_tag = st.sidebar.selectbox("Choose Tag ID:", all_tags)

# ================================================================
# HOME PAGE
# ================================================================

if page == "Home":
    st.title("🦉 Owl Migration Pattern Analysis App")
    st.write("""
    Upload your owl dataset and explore:
    - Migration patterns
    - DBSCAN clustering phases
    - Stopovers and timing
    """)

# ================================================================
# EDA PAGE
# ================================================================

elif page == "EDA":

    st.title(" Exploratory Data Analysis (EDA)")

    st.write("### Migration Summary (All Tags)")
    st.dataframe(migration_phase_summary)

    df_tag = detections[detections["Tag"] == selected_tag].copy()
    df_tag["ts"] = pd.to_datetime(df_tag["ts"], errors="coerce")
    df_tag["date"] = df_tag["ts"].dt.date

    daily = df_tag["date"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(daily.index, daily.values, marker="o")
    ax.set_title(f"Daily Detections — {selected_tag}")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# ================================================================
# MODEL PAGE
# ================================================================

elif page == "Model":

    st.title(" Migration Model (DBSCAN Phases)")
    st.dataframe(migration_phase_summary)

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
                   color=colors.get(phase, "black"),
                   s=10, label=phase)

    ax.set_title(f"Migration Phases — {selected_tag}")
    plt.xticks(rotation=45)
    ax.legend()
    st.pyplot(fig)

# ================================================================
# CHATBOT PAGE
# ================================================================

elif page == "Chatbot":

    st.title("🦉 Owl Migration Chatbot")

    def build_kb(mig, xai):
        rows = []
        for _, r in mig.iterrows():
            rows.append(
                f"Owl {r['Tag']} had {r['Num_Clusters']} clusters, arrival on {r['Arrival']}, departure on {r['Departure']}."
            )
        for _, r in xai.iterrows():
            rows.append(
                f"Owl {r['Tag']} had {r['Count']} detections during {r['Phase']} phase."
            )
        return rows

    kb = build_kb(migration_phase_summary, xai_summary)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(kb)

    question = st.text_input("Ask your question:")

    if question:
        q_vec = vectorizer.transform([question])
        sims = cosine_similarity(q_vec, X)[0]
        best_idx = sims.argmax()

        st.subheader("Answer:")
        st.write(kb[best_idx])

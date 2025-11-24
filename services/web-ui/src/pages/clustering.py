import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px
from config import API_URL, get_api_headers

def render():
    st.title("📊 Clustering")

    if "session_id" not in st.session_state:
        st.warning("Please select or create a session first.")
        return

    session_id = st.session_state["session_id"]

    if st.button("Start Clustering"):
        try:
            payload = {"session_id": session_id}
            headers = get_api_headers()
            response = requests.post(f"{API_URL}/api/clustering-service/cluster", json=payload, headers=headers)
            if response.status_code == 200:
                st.success("Clustering started.")
            else:
                st.error(f"Failed to start clustering: {response.text}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.subheader("Clusters Visualization")

    try:
        response = requests.get(f"{API_URL}/api/clustering-service/clusters/{session_id}")
        if response.status_code == 200:
            clusters = response.json()
            if clusters:
                try:
                    df = pd.DataFrame(clusters)
                    if not df.empty and 'x' in df.columns and 'y' in df.columns:
                        fig = px.scatter(
                            df,
                            x="x",
                            y="y",
                            color="label" if "label" in df.columns else None,
                            hover_data=["title"] if "title" in df.columns else None,
                            title="Cluster Visualization"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Cluster data format not recognized for visualization.")
                        st.json(clusters)
                except Exception as e:
                    st.error(f"Error visualizing clusters: {e}")
                    st.json(clusters)
            else:
                st.info("No clusters found.")
        else:
            st.info("Could not fetch cluster data.")
    except Exception as e:
        st.error(f"Error fetching clusters: {e}")

"""Clustering visualization page for Streamlit UI."""

import streamlit as st
from ..api.client import SyncAPIClient


def render_clustering_page():
    """Render the clustering visualization page."""
    st.header("🗂️ Tab Clusters")
    
    # Initialize API client
    if "api_client" not in st.session_state:
        st.session_state.api_client = SyncAPIClient()
    
    api = st.session_state.api_client
    
    # Check for current session
    if not st.session_state.get("current_session_id"):
        st.warning("Please select a session first on the URL Input page.")
        return
    
    session_id = st.session_state.current_session_id
    
    # Clustering controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔄 Generate Clusters", key="generate_clusters", type="primary"):
            with st.spinner("Clustering URLs... This may take a moment."):
                try:
                    result = api.start_clustering(session_id)
                    st.session_state.clusters = result.get("clusters", [])
                    st.success(f"Generated {len(st.session_state.clusters)} clusters")
                except Exception as e:
                    st.error(f"Clustering failed: {e}")
    
    with col2:
        if st.button("📥 Load Existing Clusters", key="load_clusters"):
            try:
                result = api.get_clusters(session_id)
                st.session_state.clusters = result.get("clusters", [])
                if st.session_state.clusters:
                    st.success(f"Loaded {len(st.session_state.clusters)} clusters")
                else:
                    st.info("No clusters found. Generate them first.")
            except Exception as e:
                st.error(f"Failed to load clusters: {e}")
    
    st.divider()
    
    # Display clusters
    clusters = st.session_state.get("clusters", [])
    
    if not clusters:
        st.info("No clusters to display. Add URLs and generate clusters.")
        return
    
    # Cluster summary
    st.subheader("Cluster Overview")
    
    total_urls = sum(c.get("tab_count", len(c.get("urls", []))) for c in clusters)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Clusters", len(clusters))
    with col2:
        st.metric("Total URLs", total_urls)
    with col3:
        avg_size = total_urls / len(clusters) if clusters else 0
        st.metric("Avg Cluster Size", f"{avg_size:.1f}")
    
    st.divider()
    
    # View mode
    view_mode = st.radio(
        "View Mode",
        ["Cards", "List", "Tree"],
        horizontal=True,
        key="cluster_view_mode",
    )
    
    if view_mode == "Cards":
        render_clusters_cards(clusters)
    elif view_mode == "List":
        render_clusters_list(clusters)
    else:
        render_clusters_tree(clusters)


def render_clusters_cards(clusters: list):
    """Render clusters as cards."""
    # Create columns for card layout
    cols = st.columns(2)
    
    for i, cluster in enumerate(clusters):
        with cols[i % 2]:
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 16px;
                        margin-bottom: 16px;
                        background: #f9f9f9;
                    ">
                        <h4 style="margin: 0 0 8px 0;">🏷️ {cluster.get('name', f'Cluster {cluster.get("id", i)}')}</h4>
                        <p style="color: #666; margin: 0;">
                            {cluster.get('tab_count', len(cluster.get('urls', [])))} URLs
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                with st.expander("View URLs"):
                    for url_data in cluster.get("urls", []):
                        url = url_data.get("url", url_data) if isinstance(url_data, dict) else url_data
                        title = url_data.get("title", url) if isinstance(url_data, dict) else url
                        st.markdown(f"- [{title[:50]}...]({url})" if len(title) > 50 else f"- [{title}]({url})")
                
                # Subclusters
                if cluster.get("subclusters"):
                    with st.expander("Subclusters"):
                        for sub in cluster["subclusters"]:
                            st.write(f"**{sub.get('name', 'Subcluster')}** ({sub.get('tab_count', 0)} URLs)")


def render_clusters_list(clusters: list):
    """Render clusters as a list."""
    for i, cluster in enumerate(clusters):
        cluster_name = cluster.get("name", f"Cluster {cluster.get('id', i)}")
        url_count = cluster.get("tab_count", len(cluster.get("urls", [])))
        
        with st.expander(f"🏷️ {cluster_name} ({url_count} URLs)", expanded=i == 0):
            # URLs table
            urls = cluster.get("urls", [])
            
            if urls:
                for url_data in urls:
                    if isinstance(url_data, dict):
                        url = url_data.get("url", "")
                        title = url_data.get("title", url)
                    else:
                        url = url_data
                        title = url_data
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"[{title[:60]}...]({url})" if len(title) > 60 else f"[{title}]({url})")
                    with col2:
                        st.caption(url[:30] + "..." if len(url) > 30 else url)
            
            # Subclusters
            if cluster.get("subclusters"):
                st.write("---")
                st.write("**Subclusters:**")
                for sub in cluster["subclusters"]:
                    st.write(f"- {sub.get('name', 'Subcluster')} ({sub.get('tab_count', 0)} URLs)")


def render_clusters_tree(clusters: list):
    """Render clusters as a tree structure."""
    for i, cluster in enumerate(clusters):
        cluster_name = cluster.get("name", f"Cluster {cluster.get('id', i)}")
        url_count = cluster.get("tab_count", len(cluster.get("urls", [])))
        
        st.markdown(f"### 📁 {cluster_name}")
        st.caption(f"{url_count} URLs")
        
        # URLs
        urls = cluster.get("urls", [])
        for url_data in urls[:10]:  # Limit display
            if isinstance(url_data, dict):
                url = url_data.get("url", "")
                title = url_data.get("title", url)
            else:
                url = url_data
                title = url_data
            
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📄 [{title[:50]}]({url})")
        
        if len(urls) > 10:
            st.caption(f"&nbsp;&nbsp;&nbsp;&nbsp;... and {len(urls) - 10} more")
        
        # Subclusters
        if cluster.get("subclusters"):
            for sub in cluster["subclusters"]:
                sub_name = sub.get("name", "Subcluster")
                sub_count = sub.get("tab_count", 0)
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;📂 **{sub_name}** ({sub_count} URLs)")
        
        st.write("")

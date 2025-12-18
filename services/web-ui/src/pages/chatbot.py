"""Chatbot page for Streamlit UI."""

import streamlit as st
from ..api.client import SyncAPIClient


def render_chatbot_page():
    """Render the chatbot page."""
    st.header("💬 Chat with Your Tabs")
    
    # Initialize API client
    if "api_client" not in st.session_state:
        st.session_state.api_client = SyncAPIClient()
    
    api = st.session_state.api_client
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Session context
    session_id = st.session_state.get("current_session_id")
    
    if session_id:
        st.caption(f"Chatting about content from current session")
    else:
        st.caption("No session selected - chatting about all indexed content")
    
    # Sidebar options
    with st.sidebar:
        st.subheader("Chat Options")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        # Quick actions
        st.subheader("Quick Actions")
        
        if st.button("📝 Summarize Session"):
            if session_id:
                with st.spinner("Generating summary..."):
                    try:
                        result = api.summarize_session(session_id)
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": "Summarize all the content in this session",
                        })
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result.get("summary", "No summary available"),
                        })
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate summary: {e}")
            else:
                st.warning("Please select a session first")
        
        if st.button("🔍 Search Mode"):
            st.session_state.chat_mode = "search"
            st.success("Switched to search mode")
        
        if st.button("💭 Chat Mode"):
            st.session_state.chat_mode = "chat"
            st.success("Switched to chat mode")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"- [{source.get('title', 'Untitled')}]({source.get('url', '#')}) "
                            f"(relevance: {source.get('score', 0):.2f})"
                        )
    
    # Chat input
    if prompt := st.chat_input("Ask about your tabs..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    mode = st.session_state.get("chat_mode", "chat")
                    
                    if mode == "search":
                        result = api.search(prompt, session_id)
                        
                        # Format search results
                        results = result.get("results", [])
                        if results:
                            response = "Here are the most relevant pages:\n\n"
                            for i, r in enumerate(results, 1):
                                response += f"{i}. **{r.get('title', 'Untitled')}**\n"
                                response += f"   {r.get('url', '')}\n"
                                response += f"   _{r.get('content', '')[:200]}..._\n\n"
                        else:
                            response = "No relevant results found."
                        
                        st.markdown(response)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                        })
                    else:
                        result = api.chat(prompt, session_id)
                        
                        answer = result.get("answer", "I couldn't generate a response.")
                        sources = result.get("sources", [])
                        
                        st.markdown(answer)
                        
                        if sources:
                            with st.expander("📚 Sources"):
                                for source in sources:
                                    st.markdown(
                                        f"- [{source.get('title', 'Untitled')}]({source.get('url', '#')}) "
                                        f"(relevance: {source.get('score', 0):.2f})"
                                    )
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                        
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
    
    # Example queries
    if not st.session_state.chat_history:
        st.divider()
        st.subheader("💡 Example Questions")
        
        examples = [
            "What are the main topics covered in my tabs?",
            "Find articles about Python programming",
            "Summarize the documentation pages",
            "What are the common themes across these pages?",
            "Which pages discuss machine learning?",
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": example,
                    })
                    st.rerun()

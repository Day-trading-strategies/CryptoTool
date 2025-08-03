import streamlit as st

def debug_navigation():
    st.title("Navigation Debug Test")
    
    # Initialize test data
    if "test_position" not in st.session_state:
        st.session_state.test_position = 50
    
    if "test_highlights" not in st.session_state:
        st.session_state.test_highlights = []
    
    st.write(f"Current Position: {st.session_state.test_position}")
    st.write(f"Current Highlights: {st.session_state.test_highlights}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("⬅️ Back"):
            if st.session_state.test_position > 0:
                st.session_state.test_position -= 1
                st.rerun()
    
    with col2:
        if st.button("➡️ Forward"):
            if st.session_state.test_position < 100:
                st.session_state.test_position += 1
                st.rerun()
    
    with col3:
        if st.button("Add Highlight"):
            st.session_state.test_highlights.append(st.session_state.test_position)
            st.rerun()
    
    with col4:
        if st.button("Clear Highlights"):
            st.session_state.test_highlights = []
            st.rerun()
    
    with col5:
        if st.button("Reset Position"):
            st.session_state.test_position = 100
            st.rerun()

if __name__ == "__main__":
    debug_navigation()

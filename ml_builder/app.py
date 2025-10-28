import streamlit as st
from Builder import Builder, ModelStage
from utils.logging.logger import MLLogger
from utils.journey_point_utils import render_journey_point_popover

st.set_page_config(
    page_title="ML Model Development Guide", 
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        #'About': "A guided ML model development application that promotes responsible AI practices"
        'About': None
    }
)

def initialize_session_state():
    """Initialize session state variables with logger."""
    if 'builder' not in st.session_state:
        st.session_state.builder = Builder()
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = ModelStage.DATA_LOADING
    if 'logger' not in st.session_state:
        st.session_state.logger = MLLogger()
    # Initialize journey tracker if not present
    if 'journey_tracker' not in st.session_state:
        from utils.logging.journey_tracker import MLJourneyTracker
        st.session_state.journey_tracker = MLJourneyTracker()

def get_next_page_name(current_stage):
    """Get the name of the next page based on current stage."""
    stage_to_page = {
        ModelStage.DATA_LOADING: "2_Data_Exploration",
        ModelStage.DATA_EXPLORATION: "3_Data_Preprocessing", 
        ModelStage.DATA_PREPROCESSING: "4_Feature_Selection",
        ModelStage.FEATURE_SELECTION: "5_Model_Selection",
        ModelStage.MODEL_SELECTION: "6_Model_Training",
        ModelStage.MODEL_TRAINING: "7_Model_Evaluation",
        ModelStage.MODEL_EVALUATION: "8_Model_Explanation",
        ModelStage.MODEL_EXPLANATION: "9_Summary",
        ModelStage.SUMMARY: "10_Predict"
    }
    return stage_to_page.get(current_stage)

def create_sidebar_navigation():
    """Create sidebar navigation with logging."""
    #st.sidebar.markdown("### Navigation")
    #stages = list(ModelStage)
    
    # Initialize session state variables if they don't exist
    #if 'current_stage' not in st.session_state:
    #    st.session_state.current_stage = ModelStage.DATA_LOADING
    #if 'builder' not in st.session_state:
    #    st.session_state.builder = Builder()
    #if 'logger' not in st.session_state:
    #    st.session_state.logger = MLLogger()
    
    #current_stage = st.session_state.current_stage
    
    #for stage in stages:
    #    page_name = get_next_page_name(stage)
    #    if page_name is None:
    #        continue
            
        # Format stage name for display
    #    display_name = stage.value.replace('_', ' ')
        
        # Log stage transitions
    #    if stage == current_stage and not hasattr(st.session_state, 'last_stage'):
    #        st.session_state.logger.log_stage_transition(
    #            "START",
    #            current_stage.value
    #        )
    #        st.session_state.last_stage = current_stage
        
        # Show completion status and make current stage prominent
    #    if st.session_state.builder.stage_completion.get(stage, False):
    #        st.sidebar.success(f"✅ {display_name}")
    #    elif stage == current_stage:
    #        st.sidebar.info(f"📍 {display_name}")
    #    else:
    #        st.sidebar.write(f"⏳ {display_name}")

    # Add reset button at the bottom of sidebar
    #st.sidebar.markdown("---")  # Add separator
    
    # Create Journey Point popover in the sidebar
    render_journey_point_popover()

    if st.sidebar.button("🔄 Reset Application"):
        # Log the reset action before clearing state
        if 'logger' in st.session_state:
            st.session_state.logger.log_user_action(
                "Application Reset"#, 
                #{"action": "reset", "from_stage": current_stage.value}
            )
            st.session_state.logger.flush_logs()
        
        # Clear all cached data
        st.cache_data.clear()
        
        # Clear session state but remember we need to reset
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Set a flag to initialize on next load
        st.session_state.needs_initialization = True
        
        # Navigate back to main app page and reset everything
        st.switch_page("app.py")

def main():
    # Check if we need to initialize after a reset
    if 'needs_initialization' in st.session_state and st.session_state.needs_initialization:
        initialize_session_state()
        del st.session_state.needs_initialization
        st.rerun()
        
    st.title("🤖 Responsible ML Model Development Guide")
    
    initialize_session_state()
    
    # Add sidebar navigation
    create_sidebar_navigation()
    
    # Introduction section
    st.markdown("""
    ## Welcome to the ML Model Development Guide!
    
    This application will guide you through the process of developing a machine learning model 
    responsibly and effectively. Each stage is designed to help you make informed decisions 
    and consider ethical implications.
    
    ### 🎯 Key Features
    - Step-by-step guidance through the ML development process
    - Built-in data analysis and visualization tools
    - Ethical AI considerations at each stage
    - Interactive model training and evaluation
    - Detailed explanations and insights
    
    ### 📋 Development Stages
    1. **Data Loading**: Upload and validate your dataset
    2. **Data Exploration**: Analyse and understand your data
    3. **Data Preprocessing**: Clean and prepare your data
    4. **Feature Selection**: Choose the most relevant features
    5. **Model Selection**: Pick the right algorithm
    6. **Model Training**: Train and tune your model
    7. **Model Evaluation**: Assess model performance
    8. **Model Explanation**: Understand model decisions
    
    ### 🚀 Getting Started
    Click the "Start ML Development" button below to begin your ML development journey.
    Each stage builds upon the previous one, ensuring a comprehensive and systematic approach.
    """)
    
    # Display current progress if any
    if st.session_state.builder.data is not None:
        st.sidebar.markdown("### Current Progress")
        stages = list(ModelStage)
        for stage in stages:
            if st.session_state.builder.stage_completion.get(stage, False):
                st.sidebar.success(f"✅ {stage.value}")
            else:
                st.sidebar.write(f"⏳ {stage.value}")

    # Add helpful resources
    with st.expander("📚 Additional Resources"):
        st.markdown("""
        - [Machine Learning Basics](https://www.geeksforgeeks.org/machine-learning/)
        - [Responsible AI Practices](https://www.ibm.com/think/topics/responsible-ai)
        - [Data Science Ethics](https://www.geeksforgeeks.org/ethics-in-data-science-and-proper-privacy-and-usage-of-data/)
        - [Feature Selection Guide](https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/)
        - [Model Evaluation Metrics](https://www.geeksforgeeks.org/machine-learning-model-evaluation/?ref=header_outind)
        """)

    st.divider()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
       st.success("✅ Start your ML journey here")
    with col2:
        st.write(" ")
    with col3:
        proceed_button = st.button(
            "Start ML Development", type="primary"
        )
        # Add navigation button to data loading page
        if proceed_button:
            next_page = "1_Data_Loading"
            st.switch_page(f"pages/{next_page}.py")
    
    # Bottom footer with version and copyright
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666666; padding: 10px;'>
        <small>Version 1.0.0 | Copyright © 2025, Richard Wheeler. All rights reserved.</small><br>
        <small>ML Model Development Guide</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
from typing import Optional
import streamlit as st
from Builder import ModelStage


def render_journey_point_popover():
    """
    Render a Streamlit popover in the sidebar to create a Journey Point (note).

    Behavior:
    - Uses current ModelStage from st.session_state
    - Saves a NOTE-type decision to the MLJourneyTracker without creating a new branch
      by attaching to the latest node in the same stage when available
    """
    # Ensure required session objects exist
    if 'journey_tracker' not in st.session_state:
        st.sidebar.info("Journey tracker not initialized yet.")
        return


    # Determine current stage from Builder if available, else fallback to session key
    if 'builder' in st.session_state and hasattr(st.session_state.builder, 'current_stage'):
        current_stage = st.session_state.builder.current_stage
    else:
        current_stage = st.session_state.get('current_stage', ModelStage.DATA_LOADING)
    # Map to the tracker stage format (UPPERCASE names)
    stage_str = current_stage.name if isinstance(current_stage, ModelStage) else str(current_stage).upper()

    # Sidebar popover to add a note
    with st.sidebar.popover("Create Journey Point", width='stretch'):
        st.caption("Add a note to your ML journey at the current stage.")
        with st.form("journey_note_form", clear_on_submit=True):
            note_text = st.text_area(
                "Note",
                key="journey_note_textarea",
                placeholder="Write a brief note...",
                height=120,
            )
            submitted = st.form_submit_button("Save", type="primary", width='stretch')

            if submitted:
                text = (note_text or "").strip()
                if not text:
                    st.warning("Please enter a note before saving.")
                else:
                    tracker = st.session_state.journey_tracker

                    # Let the tracker determine the parent automatically based on stage progression
                    # The tracker's add_decision method has logic to find the optimal parent
                    tracker.add_decision(
                        stage=stage_str,
                        decision_type="NOTE",
                        description=text,
                        details={"source": "sidebar_popover"},
                        parent_id=None,  # Let tracker auto-determine parent
                    )

                    if hasattr(st, "toast"):
                        st.toast("Journey point saved.", icon="✅")
                    else:
                        st.success("Journey point saved.")

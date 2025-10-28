import streamlit as st
from datetime import datetime
import io
import plotly.graph_objects as go
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

def _format_detail_value(value):
    """Format detail values for human-readable display."""
    if value is None:
        return "None"
    
    # Handle numpy values
    if hasattr(value, 'item'):  # numpy scalar
        value = value.item()
    
    # Handle lists
    if isinstance(value, list):
        if not value:
            return "[]"
        
        # Check if it's a list of dictionaries (like feature importances or fairness results)
        if all(isinstance(item, dict) for item in value):
            formatted_items = []
            for item in value:
                if len(item) <= 3:  # Simple dict, format inline
                    item_parts = []
                    for k, v in item.items():
                        if hasattr(v, 'item'):  # numpy scalar
                            v = v.item()
                        if isinstance(v, float):
                            v = f"{v:.4f}"
                        # Convert key to string before calling replace
                        key_str = str(k).replace('_', ' ').title() if isinstance(k, str) else str(k)
                        item_parts.append(f"{key_str}: {v}")
                    formatted_items.append(" | ".join(item_parts))
                else:  # Complex dict, format with line breaks
                    item_parts = []
                    for k, v in item.items():
                        if hasattr(v, 'item'):  # numpy scalar
                            v = v.item()
                        if isinstance(v, float):
                            v = f"{v:.4f}"
                        # Convert key to string before calling replace
                        key_str = str(k).replace('_', ' ').title() if isinstance(k, str) else str(k)
                        item_parts.append(f"  {key_str}: {v}")
                    formatted_items.append("\n".join(item_parts))
            
            if len(formatted_items) == 1:
                return formatted_items[0]
            else:
                return "\n• " + "\n• ".join(formatted_items)
        
        # Handle list of simple values
        formatted_list = []
        for item in value:
            if hasattr(item, 'item'):  # numpy scalar
                item = item.item()
            if isinstance(item, float):
                item = f"{item:.4f}"
            formatted_list.append(str(item))
        
        if len(formatted_list) <= 5:
            return ", ".join(formatted_list)
        else:
            return ", ".join(formatted_list[:5]) + f" ... (+{len(formatted_list)-5} more)"
    
    # Handle dictionaries
    if isinstance(value, dict):
        if not value:
            return "{}"
        
        formatted_dict = []
        for k, v in value.items():
            if hasattr(v, 'item'):  # numpy scalar
                v = v.item()
            if isinstance(v, float):
                v = f"{v:.4f}"
            elif isinstance(v, bool):
                v = "Yes" if v else "No"
            # Convert key to string before calling replace
            key_str = str(k).replace('_', ' ').title() if isinstance(k, str) else str(k)
            formatted_dict.append(f"{key_str}: {v}")
        
        if len(formatted_dict) <= 3:
            return " | ".join(formatted_dict)
        else:
            return "\n• " + "\n• ".join(formatted_dict)
    
    # Handle floats
    if isinstance(value, float):
        if abs(value) < 0.001 or abs(value) > 1000:
            return f"{value:.2e}"
        else:
            return f"{value:.4f}"
    
    # Handle booleans
    if isinstance(value, bool):
        return "Yes" if value else "No"
    
    # Handle strings and other types
    return str(value)

def _get_branch_colors(branch_number):
    """Get background and border colors for a branch number that match the journey map colors."""
    if branch_number == 0:
        # Main path colors (blue)
        return {
            'bg_color': 'rgba(46, 134, 171, 0.08)',  # Light blue background
            'border_color': 'rgba(46, 134, 171, 0.6)',  # Blue border
        }
    else:
        # Branch colors (cycling through the same colors as journey map)
        branch_bg_colors = [
            'rgba(242, 66, 54, 0.08)',   # Light red
            'rgba(255, 165, 0, 0.08)',   # Light orange  
            'rgba(138, 43, 226, 0.08)',  # Light purple
            'rgba(34, 139, 34, 0.08)',   # Light green
            'rgba(255, 20, 147, 0.08)',  # Light pink
        ]
        branch_border_colors = [
            'rgba(242, 66, 54, 0.6)',
            'rgba(255, 165, 0, 0.6)',
            'rgba(138, 43, 226, 0.6)',
            'rgba(34, 139, 34, 0.6)',
            'rgba(255, 20, 147, 0.6)',
        ]
        
        # Use modulo to cycle through colors for higher branch numbers
        color_idx = (branch_number - 1) % len(branch_bg_colors)
        
        return {
            'bg_color': branch_bg_colors[color_idx],
            'border_color': branch_border_colors[color_idx],
        }

def render_journey_viewer(expanded=False):
    """Render the ML journey visualization component with enhanced interactivity."""
    
    # Add enhanced CSS for journey viewer
    st.markdown("""
        <style>
        /* Journey viewer specific styles */
        .journey-viewer-container .stButton > button {
            width: 100%;
            margin: 0;
            padding: 0.5rem 1rem;
        }
        .journey-viewer-container .stSelectbox {
            margin-bottom: 0;
        }
        .journey-viewer-container [data-testid="column"] {
            padding: 0.5rem;
            display: flex;
            align-items: center;
        }
        
        /* Story timeline styles */
        .story-timeline {
            background: linear-gradient(to right, #f8f9fa, #ffffff);
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .story-step {
            border-left: 4px solid #007acc;
            padding-left: 1rem;
            margin: 1rem 0;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 1rem;
        }
        
        .story-step-header {
            font-weight: bold;
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .story-step-meta {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 0.5rem;
        }
        
        .story-step-details {
            background: #f8f9fa;
            padding: 0.8rem;
            border-radius: 5px;
            margin-top: 0.5rem;
            border: 1px solid #e9ecef;
        }
        
        .stage-filter-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1rem 0;
        }
        
        .stage-filter-button {
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            border: 1px solid #ddd;
            background: white;
            cursor: pointer;
            font-size: 0.8em;
        }
        
        .stage-filter-button.active {
            background: #007acc;
            color: white;
            border-color: #007acc;
        }
        
        .details-container {
            margin-top: 0.5rem;
            padding: 0.8rem;
            background: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
        
        .details-toggle {
            cursor: pointer;
            color: #007acc;
            text-decoration: underline;
            font-size: 0.9em;
        }
        
        /* Enhanced styling for formatted details */
        .story-step-details div {
            margin: 0.3rem 0;
            line-height: 1.4;
        }
        
        .story-step-details strong {
            color: #2c3e50;
            font-weight: 600;
        }
        
        /* Special styling for list items in details */
        .story-step-details div:has(br) {
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("🗺️ Development Journey Map", expanded=expanded):
        if 'journey_tracker' in st.session_state and st.session_state.journey_tracker.journey_points:
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["🎯 Journey Map", "📖 Development Story", "📊 Analytics"])
            
            with tab1:
                # Enhanced visualization with stage separation
                fig = st.session_state.journey_tracker.generate_visualization(include_annotations=False)
                
                # Add click interactivity info with branching information
                st.info("💡 **Enhanced Branching Visualization!** The journey map now creates separate visual groups for different branch paths within each stage. Main path steps are in blue boxes with solid borders, while different branch groups get their own colored boxes with dotted borders (red, orange, purple, green, pink). Group labels show 📍 Main path and 🔀 Branch numbers when there are multiple paths in a stage. Hover over nodes for details!")
                
                # Display the enhanced visualization
                chart_container = st.plotly_chart(fig, config={'responsive': True}, key="journey_chart")
                
                # Export options
                st.markdown('<div class="journey-viewer-container">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 1, 3], vertical_alignment="bottom", gap="medium")
                with col1:
                    st.write("Export format: HTML")  # Show format but no selection needed
                
                with col2:
                    # Generate HTML file content
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"ml_journey_map_{timestamp}.html"
                    
                    buffer = io.StringIO()
                    fig.write_html(buffer)
                    data = buffer.getvalue()
                    
                    st.download_button(
                        label="Export Journey Map",
                        data=data,
                        file_name=filename,
                        mime="text/html",
                        width='stretch',
                        type="primary"
                    )
                with col3:
                    st.write(" ")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                # Interactive Development Story
                st.markdown("### 📖 Your ML Development Story")
                
                # Add explanation of branching (using regular section instead of nested expander)
                if st.checkbox("ℹ️ Show Branching Guide", value=False, key="branching_guide"):
                    st.markdown("""
                    **What is Branching?**
                    
                    Your ML development journey now supports **branching** - when you go back to earlier stages and make different decisions, the system creates a new branch rather than overwriting your previous work.
                    
                    **Visual Indicators:**
                    - 📍 **Main Path Steps** (solid border): Your initial progression through stages
                    - 🔀 **Branch Steps** (dashed border): Alternative decisions made when revisiting stages
                    
                    **Common Branching Scenarios:**
                    - Going back to **Model Selection** after seeing evaluation results to try a different algorithm
                    - Returning to **Feature Selection** to remove low-importance features
                    - Revisiting **Data Preprocessing** to try different scaling or encoding methods
                    - Making changes to **Data Exploration** based on later insights
                    
                    **Benefits:**
                    - **Complete History**: See all your decision-making, not just the final path
                    - **Learning Insights**: Understand what worked and what didn't
                    - **Reproducibility**: Full documentation of your iterative process
                    - **Team Sharing**: Show colleagues your complete thought process
                    """)
                    st.divider()
                
                # Get the story points
                story_points = st.session_state.journey_tracker.get_journey_story()
                
                if story_points:
                    # Stage filter
                    all_stages = list(set([point['stage'] for point in story_points]))
                    selected_stages = st.multiselect(
                        "Filter by stages:",
                        all_stages,
                        default=all_stages,
                        key="story_stage_filter"
                    )
                    
                    # Filter story points
                    filtered_points = [point for point in story_points if point['stage'] in selected_stages]
                    
                    # Display story timeline
                    st.markdown('<div class="story-timeline">', unsafe_allow_html=True)
                    
                    for point in filtered_points:
                        # Create story step with content inside the container
                        stage_color = point['stage_color']
                        
                        # Get consistent colors based on branch number
                        branch_number = point.get('branch_number', 0)
                        colors = _get_branch_colors(branch_number)
                        
                        # Determine styling based on whether this is a branch
                        is_branch = point.get('is_branch', False)
                        border_style = "dashed" if is_branch else "solid"
                        border_color = colors['border_color']
                        bg_color = colors['bg_color']
                        icon = "🔀" if is_branch else "📍"
                        
                        # Build technical details HTML
                        details_html = ""
                        if point['details']:
                            for key, value in point['details'].items():
                                formatted_value = _format_detail_value(value)
                                # Convert newlines to HTML line breaks for proper display
                                formatted_value = formatted_value.replace('\n', '<br>')
                                details_html += f"<div>• <strong>{key.replace('_', ' ').title()}:</strong> {formatted_value}</div>"
                        else:
                            details_html = "<div><em>No technical details recorded for this step.</em></div>"
                        
                        st.markdown(f"""
                        <div class="story-step" style="border-left-color: {border_color}; border-left-style: {border_style}; background: {bg_color};">
                            <div class="story-step-header">
                                {icon} Step {point['step_number']}: {point['decision_type']}
                            </div>
                            <div class="story-step-meta">
                                🏷️ <strong>{point['stage']}</strong> | 
                                🕒 {point['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                                {' | 🔀 <strong>Branch Decision</strong>' if is_branch else ' | 📍 <strong>Main Path</strong>'}
                            </div>
                            <div style="display: flex; gap: 1rem; margin-top: 0.8rem;">
                                <div style="flex: 1;">
                                    <div style="font-weight: bold; margin-bottom: 0.5rem;">📝 What Happened:</div>
                                    <div>{point['description']}</div>
                                </div>
                                <div style="flex: 1;">
                                    <div style="font-weight: bold; margin-bottom: 0.5rem;">🔧 Technical Details:</div>
                                    {details_html}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add some spacing between steps
                        st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced download section with format options
                    st.markdown("### 📥 Export Development Story")
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        # Format selection
                        export_format = st.selectbox(
                            "Export format:",
                            ["Text (.txt)", "PDF (.pdf)"],
                            key="export_format_selector"
                        )
                    
                    with col2:
                        # Single download button that adapts based on format
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        if export_format == "Text (.txt)":
                            # Generate text content
                            story_text = _generate_story_export(story_points)
                            filename = f"ml_development_story_{timestamp}.txt"
                            mime_type = "text/plain"
                            data = story_text
                            button_label = "📄 Download Story"
                            button_type = "primary"
                        else:  # PDF format
                            try:
                                # Generate PDF content
                                story_pdf = _generate_story_pdf(story_points)
                                filename = f"ml_development_story_{timestamp}.pdf"
                                mime_type = "application/pdf"
                                data = story_pdf
                                button_label = "📑 Download Story"
                                button_type = "primary"
                            except Exception as e:
                                st.error(f"PDF generation failed: {str(e)}")
                                st.info("💡 Make sure reportlab is installed: `pip install reportlab>=4.0.0`")
                                data = None
                        
                        if data is not None:
                            st.download_button(
                                label=button_label,
                                data=data,
                                file_name=filename,
                                mime=mime_type,
                                type=button_type,
                                width='stretch'
                            )
                    
                    with col3:
                        st.write(" ")  # Empty column for spacing
                else:
                    st.info("No journey points recorded yet. Start building your ML model to see your development story!")
            
            with tab3:
                # Analytics and insights
                st.markdown("### 📊 Journey Analytics")
                
                if story_points:
                    # Basic statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Calculate branching statistics
                    total_steps = len(story_points)
                    branch_steps = sum(1 for point in story_points if point.get('is_branch', False))
                    main_path_steps = total_steps - branch_steps
                    stages_covered = len(set([point['stage'] for point in story_points]))
                    
                    with col1:
                        st.metric("Total Steps", total_steps)
                    
                    with col2:
                        st.metric("Stages Covered", f"{stages_covered}/7")
                    
                    with col3:
                        if len(story_points) > 1:
                            duration = story_points[-1]['timestamp'] - story_points[0]['timestamp']
                            st.metric("Total Duration", f"{duration.total_seconds()/60:.1f} min")
                        else:
                            st.metric("Total Duration", "< 1 min")
                    
                    with col4:
                        if branch_steps > 0:
                            branch_percentage = (branch_steps / total_steps) * 100
                            st.metric("Branching Activity", f"{branch_percentage:.1f}%", 
                                    help="Percentage of steps that were branch decisions")
                        else:
                            st.metric("Branching Activity", "0%")
                    
                    # Additional branching insights
                    if branch_steps > 0:
                        st.markdown("#### 🔀 Branching Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Main Path Steps", main_path_steps)
                            st.metric("Branch Steps", branch_steps)
                        
                        with col2:
                            # Find most revisited stage
                            stage_visits = {}
                            for point in story_points:
                                stage = point['stage']
                                visit_num = point.get('visit_number', 1)
                                stage_visits[stage] = max(stage_visits.get(stage, 1), visit_num)
                            
                            most_revisited = max(stage_visits.items(), key=lambda x: x[1])
                            if most_revisited[1] > 1:
                                st.metric("Most Revisited Stage", most_revisited[0])
                                st.metric("Times Revisited", most_revisited[1] - 1)
                            else:
                                st.write("**No stages revisited yet**")
                        
                        # Branching insights
                        st.markdown("**💡 Insights:**")
                        if branch_percentage > 30:
                            st.info("🔄 **High Iteration**: You're doing lots of experimentation - great for finding optimal solutions!")
                        elif branch_percentage > 10:
                            st.info("⚖️ **Balanced Approach**: Good mix of forward progress and refinement.")
                        else:
                            st.info("➡️ **Linear Progress**: You're moving efficiently through the development stages.")
                    
                    # Stage progression chart (updated to show branches)
                    stage_counts = {'Main Path': {}, 'Branches': {}}
                    for point in story_points:
                        stage = point['stage']
                        is_branch = point.get('is_branch', False)
                        path_type = 'Branches' if is_branch else 'Main Path'
                        stage_counts[path_type][stage] = stage_counts[path_type].get(stage, 0) + 1
                    
                    if stage_counts:
                        # Get all unique stages
                        all_stages = set()
                        for path_data in stage_counts.values():
                            all_stages.update(path_data.keys())
                        all_stages = sorted(all_stages)
                        
                        # Prepare data for grouped bar chart
                        main_path_counts = [stage_counts['Main Path'].get(stage, 0) for stage in all_stages]
                        branch_counts = [stage_counts['Branches'].get(stage, 0) for stage in all_stages]
                        
                        fig_bar = go.Figure()
                        
                        # Add main path bars
                        fig_bar.add_trace(go.Bar(
                            x=all_stages,
                            y=main_path_counts,
                            name='Main Path Steps',
                            marker_color='#2E86AB',
                            offsetgroup=1
                        ))
                        
                        # Add branch bars
                        fig_bar.add_trace(go.Bar(
                            x=all_stages,
                            y=branch_counts,
                            name='Branch Steps',
                            marker_color='#F24236',
                            offsetgroup=1
                        ))
                        
                        fig_bar.update_layout(
                            title="Steps per Stage: Main Path vs Branches",
                            xaxis_title="Stage",
                            yaxis_title="Number of Steps",
                            barmode='group',
                            showlegend=True
                        )
                        st.plotly_chart(fig_bar, config={'responsive': True})
                else:
                    st.info("No data available for analytics yet.")
        else:
            st.info("🚀 Start your ML development journey! Your decisions and progress will be automatically tracked and visualized here.")

def _generate_story_export(story_points):
    """Generate a text export of the development story."""
    story_text = "ML Development Journey Story\n"
    story_text += "=" * 50 + "\n\n"
    
    for point in story_points:
        story_text += f"Step {point['step_number']}: {point['decision_type']}\n"
        story_text += f"Stage: {point['stage']}\n"
        story_text += f"Time: {point['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        story_text += f"Description: {point['description']}\n"
        
        if point['details']:
            story_text += "Technical Details:\n"
            for key, value in point['details'].items():
                story_text += f"  - {key.replace('_', ' ').title()}: {_format_detail_value(value)}\n"
        
        story_text += "\n" + "-" * 40 + "\n\n"
    
    return story_text

def _generate_story_pdf(story_points):
    """Generate a PDF export of the development story with color-coded branches."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=inch, bottomMargin=inch)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.darkblue,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    # Base step header style - will be customized per step with colors
    base_step_header_style = ParagraphStyle(
        'BaseStepHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        spaceBefore=5,
        spaceAfter=10
    )
    
    content_style = ParagraphStyle(
        'ContentStyle',
        parent=styles['Normal'],
        fontSize=11,
        spaceBefore=5,
        spaceAfter=5,
        leftIndent=20
    )
    
    details_style = ParagraphStyle(
        'DetailsStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.darkgreen,
        leftIndent=40,
        spaceBefore=3
    )
    
    # Helper function to convert rgba colors to reportlab colors
    def rgba_to_reportlab_color(rgba_string):
        """Convert rgba color string to reportlab Color object."""
        # Extract values from rgba string like 'rgba(46, 134, 171, 0.6)'
        import re
        match = re.match(r'rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)', rgba_string)
        if match:
            r, g, b, a = match.groups()
            # reportlab uses values 0-1, not 0-255
            return colors.Color(int(r)/255.0, int(g)/255.0, int(b)/255.0, alpha=float(a))
        return colors.lightgrey
    
    # Build content
    story = []
    
    # Title
    story.append(Paragraph("ML Development Journey Story", title_style))
    story.append(Spacer(1, 20))
    
    # Summary information
    if story_points:
        total_steps = len(story_points)
        branch_steps = sum(1 for point in story_points if point.get('is_branch', False))
        stages_covered = len(set([point['stage'] for point in story_points]))
        duration = story_points[-1]['timestamp'] - story_points[0]['timestamp']
        
        summary_data = [
            ['Total Steps', str(total_steps)],
            ['Stages Covered', f"{stages_covered}/7"],
            ['Duration', f"{duration.total_seconds()/60:.1f} minutes"],
            ['Branch Steps', str(branch_steps)],
            ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 30))
    
    # Journey steps with color coding
    for point in story_points:
        # Get branch colors
        branch_number = point.get('branch_number', 0)
        branch_colors_dict = _get_branch_colors(branch_number)
        
        # Convert colors for reportlab
        bg_color = rgba_to_reportlab_color(branch_colors_dict['bg_color'])
        border_color = rgba_to_reportlab_color(branch_colors_dict['border_color'])
        
        # Create colored step header with background and border
        branch_indicator = " 🔀" if point.get('is_branch', False) else " 📍"
        step_title = f"Step {point['step_number']}: {point['decision_type']}{branch_indicator}"
        
        # Create a custom style for this step with branch colors
        step_header_style = ParagraphStyle(
            f'StepHeader_{point["step_number"]}',
            parent=base_step_header_style,
            textColor=colors.darkblue if branch_number == 0 else border_color,
            backColor=bg_color,
            borderColor=border_color,
            borderWidth=2,
            borderPadding=8,
            borderRadius=3
        )
        
        # Create colored box for step header
        step_table_data = [[Paragraph(step_title, step_header_style)]]
        step_table = Table(step_table_data, colWidths=[6*inch])
        step_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), bg_color),
            ('LINEBELOW', (0, 0), (-1, -1), 2, border_color),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        story.append(step_table)
        story.append(Spacer(1, 10))
        
        # Meta information with color accent
        branch_type = "Branch Decision" if point.get('is_branch', False) else "Main Path"
        meta_text = f"Stage: <b>{point['stage']}</b> | Time: {point['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | <b>{branch_type}</b>"
        story.append(Paragraph(meta_text, meta_style))
        
        # Description
        story.append(Paragraph(f"<b>What Happened:</b>", content_style))
        story.append(Paragraph(point['description'], content_style))
        
        # Technical details
        if point['details']:
            story.append(Paragraph(f"<b>Technical Details:</b>", content_style))
            for key, value in point['details'].items():
                formatted_value = _format_detail_value(value)
                # Clean up the formatted value for PDF (remove HTML tags if any)
                formatted_value = formatted_value.replace('<br>', '\n').replace('<strong>', '').replace('</strong>', '')
                detail_text = f"• <b>{key.replace('_', ' ').title()}:</b> {formatted_value}"
                story.append(Paragraph(detail_text, details_style))
        
        story.append(Spacer(1, 20))
    
    # Add color legend at the end
    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>Color Legend:</b>", content_style))
    
    # Create legend table - dynamically based on actual branches in story
    legend_data = [['Color', 'Path Type', 'Description']]
    
    # Collect all unique branch numbers used in the story
    branch_numbers_used = set()
    has_main_path = False
    
    for point in story_points:
        branch_number = point.get('branch_number', 0)
        if branch_number == 0:
            has_main_path = True
        else:
            branch_numbers_used.add(branch_number)
    
    # Add main path if present
    if has_main_path:
        main_colors = _get_branch_colors(0)
        legend_data.append(['', '📍 Main Path', 'Your initial progression through stages'])
    
    # Add all branches that were actually used, sorted by branch number
    for branch_num in sorted(branch_numbers_used):
        branch_colors_dict = _get_branch_colors(branch_num)
        
        # Calculate which color cycle this represents
        color_cycle = ((branch_num - 1) % 5) + 1  # 1-5 cycle
        
        # Add note about color cycling for branches beyond 5
        description = f"Alternative decision path #{branch_num}"
        if branch_num > 5:
            description += f" (reuses color cycle {color_cycle})"
        
        legend_data.append(['', f'🔀 Branch {branch_num}', description])
    
    # Only create legend if there are entries beyond the header
    if len(legend_data) > 1:
        legend_table = Table(legend_data, colWidths=[0.5*inch, 1.5*inch, 3*inch])
        legend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        # Add colored backgrounds for each entry
        row_index = 1
        
        # Color main path if present
        if has_main_path:
            main_colors = _get_branch_colors(0)
            main_bg = rgba_to_reportlab_color(main_colors['bg_color'])
            main_border = rgba_to_reportlab_color(main_colors['border_color'])
            legend_table.setStyle(TableStyle([
                ('BACKGROUND', (0, row_index), (0, row_index), main_bg),
                ('LINEBELOW', (0, row_index), (0, row_index), 2, main_border),
            ]))
            row_index += 1
        
        # Color all branch entries
        for branch_num in sorted(branch_numbers_used):
            branch_colors_dict = _get_branch_colors(branch_num)
            bg_color = rgba_to_reportlab_color(branch_colors_dict['bg_color'])
            border_color = rgba_to_reportlab_color(branch_colors_dict['border_color'])
            legend_table.setStyle(TableStyle([
                ('BACKGROUND', (0, row_index), (0, row_index), bg_color),
                ('LINEBELOW', (0, row_index), (0, row_index), 2, border_color),
            ]))
            row_index += 1
        
        story.append(legend_table)
        
        # Add explanation about color cycling if there are many branches
        if max(branch_numbers_used, default=0) > 5:
            story.append(Spacer(1, 10))
            cycling_explanation = (
                "<i>Note: The system uses 5 distinct branch colors that cycle for branches beyond #5. "
                "For example, Branch 6 uses the same color as Branch 1, Branch 7 uses the same color as Branch 2, etc.</i>"
            )
            story.append(Paragraph(cycling_explanation, meta_style))
    else:
        # If only main path, add a simple note
        story.append(Paragraph("<i>This journey contains only main path steps (no branches taken).</i>", meta_style))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

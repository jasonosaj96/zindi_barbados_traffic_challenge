#!/usr/bin/env python3
"""
Streamlit Debug App for Barbados Traffic Challenge

This app allows you to:
- Browse and visualize processed videos
- View detection results and JSON outputs
- Inspect vehicle journeys and tracking data
- Debug zone configurations and counts
- Compare annotated vs original videos
"""

import streamlit as st
import cv2
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Traffic Analysis Debug Tool",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


class TrafficDebugApp:
    """Main application class for traffic debugging."""

    def __init__(self, data_dir: str = "video_processed_files"):
        self.data_dir = Path(data_dir)
        self.cameras = ["normanniles1", "normanniles2", "normanniles3", "normanniles4"]

    def find_videos(self, camera: Optional[str] = None, annotated_only: bool = False) -> List[Path]:
        """Find all video files in the data directory."""
        videos = []

        if camera:
            camera_dir = self.data_dir / camera
            if camera_dir.exists():
                videos = list(camera_dir.glob("*.mp4"))
        else:
            for cam in self.cameras:
                camera_dir = self.data_dir / cam
                if camera_dir.exists():
                    videos.extend(list(camera_dir.glob("*.mp4")))

        # Filter to only annotated videos if requested
        if annotated_only:
            videos = [v for v in videos if not v.name.endswith('.annotated.mp4') and
                     v.with_suffix('').with_suffix('.annotated.mp4').exists()]
        else:
            # Exclude .annotated.mp4 files from the list (only show originals)
            videos = [v for v in videos if not v.name.endswith('.annotated.mp4')]

        return sorted(videos)

    def find_json_results(self, video_path: Path) -> Optional[Path]:
        """Find corresponding JSON results file for a video."""
        json_path = video_path.with_suffix('.counts.json')
        return json_path if json_path.exists() else None

    def find_durations_json(self, video_path: Path) -> Optional[Path]:
        """Find corresponding durations JSON file for a video."""
        durations_path = video_path.with_suffix('.durations.json')
        return durations_path if durations_path.exists() else None

    def load_json(self, json_path: Path) -> Optional[Dict]:
        """Load JSON file."""
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading JSON: {e}")
            return None

    def extract_video_frame(self, video_path: Path, frame_num: int = 0) -> Optional[np.ndarray]:
        """Extract a specific frame from video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except Exception as e:
            st.error(f"Error extracting frame: {e}")
            return None

    def get_video_info(self, video_path: Path) -> Dict:
        """Get video metadata."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            info = {
                'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
            }
            cap.release()
            return info
        except Exception as e:
            st.error(f"Error getting video info: {e}")
            return {}


def render_sidebar(app: TrafficDebugApp):
    """Render sidebar with navigation and filters."""
    st.sidebar.markdown("## 🚗 Traffic Debug Tool")
    st.sidebar.markdown("---")

    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode",
        ["📹 Video Browser", "📊 Results Viewer", "🎯 Vehicle Tracking", "📈 Analytics Dashboard"],
        index=0
    )

    st.sidebar.markdown("---")

    # Camera filter
    selected_camera = st.sidebar.selectbox(
        "Select Camera",
        ["All Cameras"] + app.cameras,
        index=0
    )

    if selected_camera == "All Cameras":
        selected_camera = None

    return mode, selected_camera


def render_video_browser(app: TrafficDebugApp, camera: Optional[str]):
    """Render video browser interface."""
    st.markdown('<p class="main-header">📹 Video Browser</p>', unsafe_allow_html=True)

    # Option to show only annotated videos
    show_annotated_only = st.checkbox("Show only processed videos (with annotations)", value=True)

    # Find videos
    videos = app.find_videos(camera, annotated_only=show_annotated_only)

    if not videos:
        if show_annotated_only:
            st.warning("No annotated videos found. Please process videos with --save-video flag.")
            st.info("💡 Uncheck 'Show only processed videos' to see all available videos.")
        else:
            st.warning("No videos found. Please download videos using the automated pipeline.")
        return

    st.info(f"Found {len(videos)} videos" + (" with annotations" if show_annotated_only else ""))

    # Video selection
    video_names = [f"{v.parent.name}/{v.name}" for v in videos]
    selected_idx = st.selectbox(
        "Select Video",
        range(len(videos)),
        format_func=lambda i: video_names[i]
    )

    if selected_idx is None:
        return

    video_path = videos[selected_idx]

    # Display video info
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### 📁 {video_path.name}")

        # Get video info
        video_info = app.get_video_info(video_path)

        if video_info:
            metrics_cols = st.columns(5)
            metrics_cols[0].metric("FPS", video_info.get('fps', 'N/A'))
            metrics_cols[1].metric("Frames", video_info.get('total_frames', 'N/A'))
            metrics_cols[2].metric("Duration", f"{video_info.get('duration_sec', 0)}s")
            metrics_cols[3].metric("Width", video_info.get('width', 'N/A'))
            metrics_cols[4].metric("Height", video_info.get('height', 'N/A'))

        # Video playback options
        st.markdown("### 🎬 Video Playback")

        playback_mode = st.radio(
            "Playback Mode",
            ["Video Player", "Frame-by-Frame"],
            horizontal=True
        )

        if playback_mode == "Video Player":
            # Check for annotated video
            annotated_path = video_path.with_suffix('').with_suffix('.annotated.mp4')

            if annotated_path.exists():
                st.markdown("**🎥 Annotated Video (with detections)**")
                st.info("📍 Showing processed video with vehicle detections, tracking, and zone overlays")

                try:
                    # Display annotated video
                    st.video(str(annotated_path), format="video/mp4", muted=True)

                except Exception as e:
                    st.error(f"Error loading video: {e}")
                    st.warning("⚠️ Video playback error. This may be due to moov atom placement.")
                    st.info("💡 Try Frame-by-Frame mode or re-process the video with proper encoding.")

            else:
                st.warning("⚠️ No annotated video available")
                st.info("💡 Process videos with `--save-video` flag to generate annotated videos:")
                st.code("python src/processes/parallel_process.py --workers 4 --save-video")

                # Optionally show original video
                if st.checkbox("Show original video (may have playback issues)"):
                    try:
                        st.video(str(video_path), format="video/mp4", muted=True)
                    except Exception as e:
                        st.error(f"Error loading video: {e}")
                        st.info("💡 Original videos may not stream properly. Use Frame-by-Frame mode instead.")

        else:  # Frame-by-Frame mode
            if video_info:
                frame_num = st.slider(
                    "Select Frame",
                    0,
                    max(0, video_info.get('total_frames', 1) - 1),
                    0,
                    help="Slide to view different frames"
                )

                # Extract and display frame
                frame = app.extract_video_frame(video_path, frame_num)
                if frame is not None:
                    st.image(frame, caption=f"Frame {frame_num}", use_column_width=True)

    with col2:
        st.markdown("### 📊 Processing Results")

        # Check for JSON results
        json_path = app.find_json_results(video_path)
        durations_path = app.find_durations_json(video_path)

        if json_path:
            st.success(f"✓ Results file found")
            if st.button("View Results JSON"):
                st.session_state['show_json'] = True
        else:
            st.warning("⚠ No results file (.counts.json)")

        if durations_path:
            st.success(f"✓ Durations file found")
            if st.button("View Durations JSON"):
                st.session_state['show_durations'] = True
        else:
            st.warning("⚠ No durations file (.durations.json)")

        # Check for annotated video
        annotated_path = video_path.with_suffix('').with_suffix('.annotated.mp4')
        if annotated_path.exists():
            st.success(f"✓ Annotated video found")
            st.info(f"Size: {annotated_path.stat().st_size / 1_000_000:.2f} MB")
        else:
            st.info("ℹ No annotated video (use --save-video)")

    # Display JSON results if requested
    if st.session_state.get('show_json') and json_path:
        st.markdown("---")
        st.markdown("### 📄 Results JSON")
        data = app.load_json(json_path)
        if data:
            st.json(data)


def render_results_viewer(app: TrafficDebugApp, camera: Optional[str]):
    """Render results viewer with detailed metrics."""
    st.markdown('<p class="main-header">📊 Results Viewer</p>', unsafe_allow_html=True)

    # Find videos with results
    videos = app.find_videos(camera)
    videos_with_results = [v for v in videos if app.find_json_results(v)]

    if not videos_with_results:
        st.warning("No processed results found. Please run the processing pipeline first.")
        return

    st.info(f"Found {len(videos_with_results)} videos with results")

    # Video selection
    video_names = [f"{v.parent.name}/{v.name}" for v in videos_with_results]
    selected_idx = st.selectbox(
        "Select Video",
        range(len(videos_with_results)),
        format_func=lambda i: video_names[i]
    )

    if selected_idx is None:
        return

    video_path = videos_with_results[selected_idx]
    json_path = app.find_json_results(video_path)

    if not json_path:
        return

    # Load results
    results = app.load_json(json_path)
    if not results:
        return

    # Display metadata
    st.markdown(f"### 📁 {video_path.name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Camera", results.get('camera', 'N/A'))
    col2.metric("Date", results.get('date', 'N/A'))
    col3.metric("Time", results.get('time', 'N/A'))
    col4.metric("Duration", f"{results.get('duration_seconds', 0):.1f}s")

    st.markdown("---")

    # Vehicle tracking summary
    st.markdown("### 🚗 Vehicle Tracking Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tracked", results.get('total_vehicles_tracked', 0))
    col2.metric("Raw Detections", results.get('total_vehicles_detected_raw', 0))
    col3.metric("After Dedup", results.get('total_vehicles_after_dedup', 0))
    col4.metric("Tracks Merged", results.get('tracks_merged', 0))

    # Vehicle classes
    st.markdown("### 🚙 Vehicle Classes")
    vehicle_classes = {k.replace('vehicle_class_', ''): v
                      for k, v in results.items()
                      if k.startswith('vehicle_class_')}

    if vehicle_classes:
        class_df = pd.DataFrame(list(vehicle_classes.items()), columns=['Class', 'Count'])

        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(class_df, use_container_width=True)

        with col2:
            fig = px.pie(class_df, values='Count', names='Class',
                        title='Vehicle Class Distribution')
            st.plotly_chart(fig, use_container_width=True)

    # Zone statistics
    st.markdown("---")
    st.markdown("### 📍 Zone Statistics")

    # Extract zone data
    zone_counts = {k.replace('zone_count_', ''): v
                  for k, v in results.items()
                  if k.startswith('zone_count_')}

    if zone_counts:
        zone_df = pd.DataFrame(list(zone_counts.items()), columns=['Zone', 'Count'])

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(zone_df, use_container_width=True)

        with col2:
            fig = px.bar(zone_df, x='Zone', y='Count',
                        title='Vehicle Counts by Zone',
                        color='Count',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

    # Dwell time statistics
    st.markdown("---")
    st.markdown("### ⏱️ Dwell Time Statistics")

    dwell_data = []
    for key, value in results.items():
        if key.startswith('dwell_mean_'):
            zone = key.replace('dwell_mean_', '')
            dwell_data.append({
                'Zone': zone,
                'Mean (s)': value,
                'Median (s)': results.get(f'dwell_median_{zone}', 0),
                'Min (s)': results.get(f'dwell_min_{zone}', 0),
                'Max (s)': results.get(f'dwell_max_{zone}', 0),
                'Std (s)': results.get(f'dwell_std_{zone}', 0),
                'Count': results.get(f'dwell_count_{zone}', 0)
            })

    if dwell_data:
        dwell_df = pd.DataFrame(dwell_data)
        st.dataframe(dwell_df, use_container_width=True)

        # Plot dwell times
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dwell_df['Zone'], y=dwell_df['Mean (s)'],
                            name='Mean', error_y=dict(type='data', array=dwell_df['Std (s)'])))
        fig.update_layout(title='Average Dwell Time by Zone',
                         xaxis_title='Zone', yaxis_title='Time (seconds)')
        st.plotly_chart(fig, use_container_width=True)


def render_vehicle_tracking(app: TrafficDebugApp, camera: Optional[str]):
    """Render vehicle tracking analysis."""
    st.markdown('<p class="main-header">🎯 Vehicle Tracking Analysis</p>', unsafe_allow_html=True)

    # Find videos with results
    videos = app.find_videos(camera)
    videos_with_results = [v for v in videos if app.find_json_results(v)]

    if not videos_with_results:
        st.warning("No processed results found.")
        return

    # Video selection
    video_names = [f"{v.parent.name}/{v.name}" for v in videos_with_results]
    selected_idx = st.selectbox(
        "Select Video",
        range(len(videos_with_results)),
        format_func=lambda i: video_names[i]
    )

    if selected_idx is None:
        return

    video_path = videos_with_results[selected_idx]
    json_path = app.find_json_results(video_path)

    if not json_path:
        return

    results = app.load_json(json_path)
    if not results or 'vehicle_journeys' not in results:
        st.warning("No vehicle journey data found in results.")
        return

    journeys = results['vehicle_journeys']

    st.markdown(f"### 🚗 {len(journeys)} Vehicle Journeys")

    # Convert to DataFrame for easier analysis
    journey_summary = []
    for journey in journeys:
        journey_summary.append({
            'Vehicle ID': journey.get('vehicle_id'),
            'Class': journey.get('class_name', 'Unknown'),
            'First Seen (s)': journey.get('first_seen_time', 0),
            'Last Seen (s)': journey.get('last_seen_time', 0),
            'Duration (s)': journey.get('total_time_visible', 0),
            'Frames': journey.get('total_frames_tracked', 0),
            'Zones Visited': journey.get('num_zones_visited', 0),
            'Avg Speed (px/s)': journey.get('speed_stats', {}).get('mean', 0)
        })

    if journey_summary:
        journey_df = pd.DataFrame(journey_summary)

        # Display table
        st.dataframe(journey_df, use_container_width=True)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Duration", f"{journey_df['Duration (s)'].mean():.2f}s")
        col2.metric("Avg Frames", f"{journey_df['Frames'].mean():.0f}")
        col3.metric("Avg Zones", f"{journey_df['Zones Visited'].mean():.1f}")
        col4.metric("Avg Speed", f"{journey_df['Avg Speed (px/s)'].mean():.1f}")

        # Visualizations
        st.markdown("### 📊 Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(journey_df, x='Duration (s)',
                             title='Distribution of Journey Durations',
                             nbins=20)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(journey_df, x='Duration (s)', y='Avg Speed (px/s)',
                           color='Class', size='Frames',
                           title='Speed vs Duration',
                           hover_data=['Vehicle ID'])
            st.plotly_chart(fig, use_container_width=True)


def render_analytics_dashboard(app: TrafficDebugApp, camera: Optional[str]):
    """Render analytics dashboard with consolidated stats."""
    st.markdown('<p class="main-header">📈 Analytics Dashboard</p>', unsafe_allow_html=True)

    # Check for consolidated results
    consolidated_path = Path("consolidated_results.csv")

    if not consolidated_path.exists():
        st.warning("No consolidated results found. Run the pipeline to generate consolidated_results.csv")
        return

    # Load consolidated data
    try:
        df = pd.read_csv(consolidated_path)
        st.success(f"Loaded {len(df)} records from consolidated results")
    except Exception as e:
        st.error(f"Error loading consolidated results: {e}")
        return

    # Filter by camera if selected
    if camera:
        df = df[df['camera'] == camera]
        st.info(f"Filtered to {len(df)} records for {camera}")

    # Overall metrics
    st.markdown("### 📊 Overall Statistics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Videos", len(df))
    col2.metric("Total Vehicles", df['total_vehicles_tracked'].sum())
    col3.metric("Avg Vehicles/Video", f"{df['total_vehicles_tracked'].mean():.1f}")
    col4.metric("Cameras", df['camera'].nunique())

    # Time-based analysis
    st.markdown("---")
    st.markdown("### ⏰ Temporal Analysis")

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date

        # Vehicles per day
        daily_counts = df.groupby('date')['total_vehicles_tracked'].sum().reset_index()

        fig = px.line(daily_counts, x='date', y='total_vehicles_tracked',
                     title='Vehicle Counts Over Time',
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # Camera comparison
    st.markdown("---")
    st.markdown("### 📹 Camera Comparison")

    camera_stats = df.groupby('camera').agg({
        'total_vehicles_tracked': 'sum',
        'filename': 'count'
    }).reset_index()
    camera_stats.columns = ['Camera', 'Total Vehicles', 'Video Count']

    col1, col2 = st.columns(2)

    with col1:
        st.dataframe(camera_stats, use_container_width=True)

    with col2:
        fig = px.bar(camera_stats, x='Camera', y='Total Vehicles',
                    title='Total Vehicles by Camera',
                    color='Total Vehicles',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    # Download data
    st.markdown("---")
    st.markdown("### 💾 Export Data")

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name=f"traffic_data_{camera if camera else 'all'}.csv",
        mime="text/csv"
    )


def main():
    """Main application entry point."""

    # Initialize app
    app = TrafficDebugApp()

    # Render sidebar
    mode, selected_camera = render_sidebar(app)

    # Render selected mode
    if mode == "📹 Video Browser":
        render_video_browser(app, selected_camera)
    elif mode == "📊 Results Viewer":
        render_results_viewer(app, selected_camera)
    elif mode == "🎯 Vehicle Tracking":
        render_vehicle_tracking(app, selected_camera)
    elif mode == "📈 Analytics Dashboard":
        render_analytics_dashboard(app, selected_camera)


if __name__ == "__main__":
    main()

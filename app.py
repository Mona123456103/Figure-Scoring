#!/usr/bin/env python3
"""
Barracuda Above-Water Tracker — Web App
===========================================================================
Streamlit front-end for tracker_core.py (RTMPose above-water tracker).
Upload a video, process it, download the annotated video + tracking CSV.

Run locally:
    streamlit run app.py

Deploy: push this folder to a Hugging Face Space (Streamlit SDK).
"""

import streamlit as st
import tempfile
import os
import time
from pathlib import Path

import tracker_core as tc

st.set_page_config(
    page_title="Barracuda Above-Water Tracker",
    page_icon="🏊",
    layout="centered",
)

st.title("🏊 Barracuda Above-Water Tracker")
st.caption(
    "Upload an above-water synchronized swimming video to get pose tracking, "
    "waterline detection, and a downloadable data file."
)

# ── Sidebar settings ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    speed_choice = st.select_slider(
        "Speed vs. accuracy",
        options=["Fast", "Balanced", "Most Accurate"],
        value="Balanced",
        help=(
            "Fast: quickest results, good for previewing.\n"
            "Balanced: recommended default.\n"
            "Most Accurate: best joint precision, slowest (can take several "
            "minutes per video on this free server)."
        ),
    )
    speed_map = {
        "Fast": dict(mode="lightweight", det_frequency=4),
        "Balanced": dict(mode="balanced", det_frequency=2),
        "Most Accurate": dict(mode="performance", det_frequency=1),
    }
    chosen = speed_map[speed_choice]

    st.divider()

    manual_waterline = st.checkbox("Set waterline manually", value=False)
    waterline_value = None
    if manual_waterline:
        waterline_value = st.slider(
            "Waterline position (fraction from top of frame)",
            min_value=0.30, max_value=0.95, value=0.70, step=0.01,
        )

    st.divider()
    max_duration = st.slider(
        "Max figure duration to track (seconds)",
        min_value=10, max_value=90, value=60, step=5,
        help="Processing stops after this many seconds to keep runtimes reasonable.",
    )

    st.divider()
    st.caption(
        "⏱️ This app runs on shared CPU hardware. A 30–60 second clip can "
        "take a few minutes to process, especially on 'Most Accurate'."
    )

# ── Main upload area ──────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your video",
    type=["mp4", "mov", "m4v", "avi"],
    help="Above-water footage of a swimmer performing a figure.",
)

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📁 {uploaded_file.name} ({file_size_mb:.1f} MB)")

    MAX_SIZE_MB = 300
    if file_size_mb > MAX_SIZE_MB:
        st.error(
            f"File is too large ({file_size_mb:.0f} MB). "
            f"Please upload a video under {MAX_SIZE_MB} MB."
        )
    else:
        run_button = st.button("🚀 Process Video", type="primary", use_container_width=True)

        if run_button:
            # ── Save upload to a temp working directory ──
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_path = Path(tmp_dir) / uploaded_file.name
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                output_path = Path(tmp_dir) / f"{input_path.stem}_POSE_tracking.mp4"

                # Apply the duration limit setting to the shared module config
                tc.MAX_FIGURE_DURATION = max_duration

                progress_bar = st.progress(0.0, text="Starting...")
                status_text = st.empty()
                start_time = time.time()

                def update_progress(frame_count, total_frames):
                    if total_frames > 0:
                        pct = min(frame_count / total_frames, 1.0)
                        elapsed = time.time() - start_time
                        fps_proc = frame_count / elapsed if elapsed > 0 else 0
                        eta = (total_frames - frame_count) / fps_proc if fps_proc > 0 else 0
                        progress_bar.progress(
                            pct,
                            text=f"Processing frame {frame_count}/{total_frames} "
                                 f"(~{eta:.0f}s remaining)",
                        )

                try:
                    with st.spinner("Loading pose model (first run downloads it, ~1 min)..."):
                        video_file, csv_file = tc.process_video(
                            str(input_path),
                            output_path,
                            waterline_value,
                            mode=chosen["mode"],
                            det_frequency=chosen["det_frequency"],
                            progress_callback=update_progress,
                        )

                    if video_file is None:
                        st.error(
                            "Waterline detection failed — try setting the waterline "
                            "manually in the sidebar and re-running."
                        )
                    else:
                        progress_bar.progress(1.0, text="Done!")
                        kalman_csv = tc.apply_kalman_filter_to_csv(csv_file)

                        st.success("✅ Processing complete!")

                        # Show the annotated video
                        st.video(str(video_file))

                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            with open(video_file, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Annotated Video",
                                    data=f.read(),
                                    file_name=Path(video_file).name,
                                    mime="video/mp4",
                                    use_container_width=True,
                                )
                        with col2:
                            with open(kalman_csv, "rb") as f:
                                st.download_button(
                                    "⬇️ Download Tracking Data (CSV)",
                                    data=f.read(),
                                    file_name=Path(kalman_csv).name,
                                    mime="text/csv",
                                    use_container_width=True,
                                )

                except Exception as e:
                    st.error(f"Something went wrong during processing: {e}")
                    st.exception(e)

st.divider()
with st.expander("ℹ️ About this tracker"):
    st.markdown(
        """
        This tool uses **RTMPose-x** (via `rtmlib`) for pose detection, combined with:
        - Physical tent/background masking (top of frame blacked out before detection)
        - Blue water color validation to reject tents and pool deck
        - Automatic waterline detection (averaged from head, shoulder, and hip position)
        - Swimmer locking to keep tracking on the same person across frames
        - Multi-level smoothing and Kalman filtering for stable joint positions

        Output includes an annotated video showing the detected skeleton and waterline,
        plus a CSV of per-frame joint positions for further analysis.
        """
    )

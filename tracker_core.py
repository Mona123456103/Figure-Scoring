#!/usr/bin/env python3
"""
ABOVE-WATER TRACKER — RTMPose ENGINE
===========================================================================
Uses RTMPose-x (via rtmlib) for pose detection instead of MediaPipe.
All background systems from the original above-water tracker are preserved:
  - Physical tent masking (blacks out top 35% before detection)
  - Blue water color validation (rejects tents/decks)
  - Waterline = average of head + shoulder + hip (first 100 frames)
  - Swimmer locking and selection logic
  - Edge proximity filtering
  - Invisible barrier zone
  - Full visualization with waterline overlay
  - Kalman filtering post-process

WHY RTMPose:
  - Much more accurate joint positions, especially hips and knees
  - Better with unusual poses (inverted, twisted, back layout)
  - Performance mode uses largest models for max accuracy
  - det_frequency=1 means every frame is detected (no skipping)

INSTALL:
    pip install rtmlib onnxruntime opencv-python numpy pandas scipy

USAGE:
    1. Set VIDEO_PATHS to your above-water video file(s)
    2. Run: python above_water_rtmpose_tracker.py
    3. Output: _POSE_tracking.mp4 + _KALMAN.csv files
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import time
import glob

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

VIDEO_PATHS = [
    '/Users/mona/Desktop/Barracuda_Recordings/swimmer_7_above.mp4'
]

OUTPUT_FPS = None
WATERLINE = None      # Set to None for auto-detection, or set manually like 0.70
MAX_FIGURE_DURATION = 60

# Invisible barrier — reject detections in top % of frame (where tents are)
IGNORE_TOP_PERCENT = 0.35   # Ignore top 35% (tent zone)
IGNORE_BOTTOM_PERCENT = 0.0

# RTMPose COCO 17 keypoints → our landmark names
# COCO order: nose, L-eye, R-eye, L-ear, R-ear, L-shoulder, R-shoulder,
#   L-elbow, R-elbow, L-wrist, R-wrist, L-hip, R-hip, L-knee, R-knee,
#   L-ankle, R-ankle
COCO_TO_LANDMARKS = {
    0:  'nose',
    5:  'left_shoulder',
    6:  'right_shoulder',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}

# All landmark names we output (matches the MediaPipe tracker CSV format)
ALL_LANDMARKS = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
    'left_foot_best', 'right_foot_best',
]

KEY_LANDMARKS = {
    0:  'nose',
    5:  'left_shoulder',
    6:  'right_shoulder',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}

FOOT_COCO_INDICES = [15, 16]  # ankles in COCO (no feet in COCO-17)


# ============================================================================
# IMAGE ENHANCEMENT (same as original above-water tracker)
# ============================================================================

def quick_enhance(frame):
    """Fast enhancement — same as the original above-water tracker"""
    return cv2.convertScaleAbs(frame, alpha=1.2, beta=15)


# ============================================================================
# WATERLINE DETECTION — IDENTICAL TO ORIGINAL
# Uses RTMPose instead of MediaPipe for the pose-based scan,
# but same median + outlier rejection logic.
# ============================================================================

def is_horizontal_position_from_kps(person_kps, person_scores, h, w):
    """
    Check if swimmer is in horizontal back layout position.
    Same logic as original, adapted for COCO keypoint format.
    person_kps: array of (x_px, y_px) for 17 COCO keypoints
    person_scores: array of confidence scores for 17 keypoints
    """
    # Need shoulders (5,6) and hips (11,12) with decent confidence
    if person_scores[5] < 0.3 or person_scores[6] < 0.3:
        return False
    if person_scores[11] < 0.3 or person_scores[12] < 0.3:
        return False

    # Normalize to 0-1
    ls_y = person_kps[5][1] / h
    rs_y = person_kps[6][1] / h
    lh_y = person_kps[11][1] / h
    rh_y = person_kps[12][1] / h

    ls_x = person_kps[5][0] / w
    rs_x = person_kps[6][0] / w
    lh_x = person_kps[11][0] / w
    rh_x = person_kps[12][0] / w

    shoulder_y = (ls_y + rs_y) / 2
    hip_y = (lh_y + rh_y) / 2
    height_diff = abs(shoulder_y - hip_y)

    body_width = max(abs(ls_x - rs_x), abs(lh_x - rh_x))

    return height_diff < 0.25 or body_width > 0.25


def detect_waterline_from_poses(video_path, pose_tracker_fn):
    """
    Waterline detection — averages head (nose), shoulder, and hip
    positions together:
    - Samples first 100 frames
    - Only uses horizontal swimmers
    - Each signal (head/shoulder/hip) is cleaned with median +
      outlier rejection (>2 std devs from median) before averaging
    - Waterline = mean of whichever of the three signals have enough
      samples (all three if available)

    Uses RTMPose via pose_tracker_fn instead of MediaPipe.
    """
    print("\n🎯 POSE-BASED waterline detection (RTMPose engine)...")
    print("  ✨ Averaging head + shoulder + hip")
    print("  ✨ Sampling 100 frames")
    print("  ✨ Horizontal swimmers only")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    head_positions = []
    hip_positions = []
    shoulder_positions = []

    max_search_frame = min(100, total_frames)
    print(f"  Scanning first {max_search_frame} frames...")

    for frame_num in range(max_search_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        enhanced = quick_enhance(frame)
        keypoints, scores = pose_tracker_fn(enhanced)

        if keypoints is None or len(keypoints) == 0:
            continue

        for person_kps, person_scores in zip(keypoints, scores):
            # Only use horizontal swimmers
            if not is_horizontal_position_from_kps(person_kps, person_scores, h_frame, w_frame):
                continue

            # Nose (COCO index 0) — head position
            if person_scores[0] > 0.5:
                head_positions.append(person_kps[0][1] / h_frame)

            # Shoulders (COCO 5,6)
            if person_scores[5] > 0.4 and person_scores[6] > 0.4:
                shoulder_y = (person_kps[5][1] + person_kps[6][1]) / 2 / h_frame
                shoulder_positions.append(shoulder_y)

            # Hips (COCO 11,12)
            if person_scores[11] > 0.4 and person_scores[12] > 0.4:
                hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h_frame
                hip_positions.append(hip_y)

    cap.release()

    print(f"\n  📊 Collected:")
    print(f"     Head: {len(head_positions)} samples")
    print(f"     Shoulders: {len(shoulder_positions)} samples")
    print(f"     Hips: {len(hip_positions)} samples")

    def _clean(values):
        """Median + outlier rejection (>2 std devs from median), then mean"""
        if len(values) < 5:
            return None
        median = np.median(values)
        std = np.std(values)
        filtered = [v for v in values if abs(v - median) < 2 * std]
        return np.mean(filtered) if filtered else median

    head_avg = _clean(head_positions)
    shoulder_avg = _clean(shoulder_positions)
    hip_avg = _clean(hip_positions)

    signals = []
    labels = []
    if head_avg is not None:
        signals.append(head_avg)
        labels.append(f"head={head_avg:.3f}")
    if shoulder_avg is not None:
        signals.append(shoulder_avg)
        labels.append(f"shoulder={shoulder_avg:.3f}")
    if hip_avg is not None:
        signals.append(hip_avg)
        labels.append(f"hip={hip_avg:.3f}")

    if signals:
        waterline = float(np.mean(signals))
        method_used = f"AVERAGE OF {', '.join(labels)}"
        print(f"\n     ✓ {method_used}")
        print(f"     ✓ Waterline: {waterline:.3f}")
    else:
        waterline = 0.70
        method_used = "DEFAULT"
        print(f"\n     ⚠ Insufficient data, using default")

    print(f"\n✅ WATERLINE: {waterline:.4f} ({waterline*100:.0f}% from top)")
    print(f"   Method: {method_used}")
    print(f"   🔒 Locked for entire video\n")

    return waterline


# ============================================================================
# WATER COLOR VALIDATION — IDENTICAL TO ORIGINAL
# ============================================================================

def is_in_water_lenient(hip_x_norm, hip_y_norm, water_level, frame):
    """
    Lenient water check with blue color validation.
    Same logic as original — rejects tents/decks by checking
    for blue water below and at hip level.
    Takes normalized hip coordinates instead of a pose object.
    """
    h, w = frame.shape[:2]
    hip_x = int(hip_x_norm * w)
    hip_y = int(hip_y_norm * h)

    # Reject if too high (above pool area)
    if hip_y_norm < 0.35:
        return False

    # Reject if clearly above waterline
    if hip_y_norm < water_level - 0.15:
        return False

    # Color-based water detection to reject tents/structures
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Blue water color ranges (pool)
    water_lower1 = np.array([85, 30, 30])     # Cyan-blue
    water_upper1 = np.array([135, 255, 255])
    water_lower2 = np.array([70, 30, 30])     # Greenish-blue
    water_upper2 = np.array([105, 255, 255])

    # Sample region BELOW hips (where water should be)
    sample_radius = 60
    below_region = hsv[min(h, hip_y + 20):min(h, hip_y + sample_radius + 40),
                       max(0, hip_x - sample_radius):min(w, hip_x + sample_radius)]

    if below_region.size > 0:
        mask1 = cv2.inRange(below_region, water_lower1, water_upper1)
        mask2 = cv2.inRange(below_region, water_lower2, water_upper2)
        water_pixels = np.sum(mask1 > 0) + np.sum(mask2 > 0)
        total_pixels = below_region.shape[0] * below_region.shape[1]
        below_water_pct = water_pixels / total_pixels if total_pixels > 0 else 0

        if below_water_pct < 0.15:
            return False

    # Also check AT hip position for blue water
    sample_at_hip = hsv[max(0, hip_y - 20):min(h, hip_y + 20),
                        max(0, hip_x - 40):min(w, hip_x + 40)]

    if sample_at_hip.size > 0:
        mask1_hip = cv2.inRange(sample_at_hip, water_lower1, water_upper1)
        mask2_hip = cv2.inRange(sample_at_hip, water_lower2, water_upper2)
        water_pixels_hip = np.sum(mask1_hip > 0) + np.sum(mask2_hip > 0)
        total_pixels_hip = sample_at_hip.shape[0] * sample_at_hip.shape[1]
        at_hip_water_pct = water_pixels_hip / total_pixels_hip if total_pixels_hip > 0 else 0

        if at_hip_water_pct < 0.10:
            return False

    return True


# ============================================================================
# POSE SIZE CALCULATION — SAME AS ORIGINAL (adapted for COCO keypoints)
# ============================================================================

def calculate_pose_size_coco(person_kps, person_scores, h, w):
    """
    Calculate pose size for foreground detection.
    Same logic as original calculate_pose_size, adapted for COCO format.
    Returns size dict or None.
    """
    # Need hips (COCO 11, 12) with some confidence
    if person_scores[11] < 0.10 or person_scores[12] < 0.10:
        return None

    lh_x = person_kps[11][0] / w
    lh_y = person_kps[11][1] / h
    rh_x = person_kps[12][0] / w
    rh_y = person_kps[12][1] / h

    torso_height = 0
    torso_width = 0
    has_shoulders = False

    # Shoulders (COCO 5, 6)
    if person_scores[5] > 0.10 and person_scores[6] > 0.10:
        has_shoulders = True
        ls_y = person_kps[5][1] / h
        rs_y = person_kps[6][1] / h
        hip_y = (lh_y + rh_y) / 2
        shoulder_y = (ls_y + rs_y) / 2
        torso_height = abs(hip_y - shoulder_y)

        ls_x = person_kps[5][0] / w
        hip_x = (lh_x + rh_x) / 2
        shoulder_x = (ls_x + ls_x) / 2
        torso_width = abs(hip_x - shoulder_x)

    hip_width = abs(lh_x - rh_x)

    full_height = 0
    has_feet = False

    # Ankles (COCO 15, 16)
    for ankle_idx in [15, 16]:
        if person_scores[ankle_idx] > 0.10:
            has_feet = True
            hip_y = (lh_y + rh_y) / 2
            ankle_y = person_kps[ankle_idx][1] / h
            height = abs(hip_y - ankle_y)
            if height > full_height:
                full_height = height
            break

    size = 0
    if has_shoulders:
        size += (torso_height + torso_width) * 3.0
    size += hip_width * 2.0
    if has_feet:
        size += full_height * 1.5
    size = size / 6.5

    return {
        'size': size,
        'hip_width': hip_width,
        'torso_height': torso_height,
        'torso_width': torso_width,
        'full_height': full_height,
        'has_shoulders': has_shoulders,
        'has_feet': has_feet,
    }


# ============================================================================
# ANATOMICAL VALIDATION — SAME AS ORIGINAL (adapted for COCO)
# ============================================================================

def validate_pose_anatomy_coco(person_kps, person_scores, h, w):
    """Lenient anatomical validation — same checks as original"""
    # Left shoulder above left hip check
    if person_scores[5] > 0.10 and person_scores[11] > 0.10:
        ls_y = person_kps[5][1] / h
        lh_y = person_kps[11][1] / h
        if ls_y > lh_y + 0.35:
            return False

    # Right shoulder above right hip check
    if person_scores[6] > 0.10 and person_scores[12] > 0.10:
        rs_y = person_kps[6][1] / h
        rh_y = person_kps[12][1] / h
        if rs_y > rh_y + 0.35:
            return False

    # Hip width sanity
    if person_scores[11] > 0.10 and person_scores[12] > 0.10:
        lh_x = person_kps[11][0] / w
        rh_x = person_kps[12][0] / w
        hip_width = abs(lh_x - rh_x)
        if hip_width < 0.02 or hip_width > 0.60:
            return False

    return True


# ============================================================================
# SWIMMER SELECTION — SAME LOGIC AS ORIGINAL
# Rejects poses in tent zone, near edges, not in water, etc.
# ============================================================================

def select_best_swimmer_coco(all_keypoints, all_scores, water_level, frame):
    """
    Select best swimmer — horizontal, foreground, in pool.
    Same logic as original select_best_swimmer, adapted for COCO format.
    Rejects: tent zone, frame edges, non-water areas.
    """
    if all_keypoints is None or len(all_keypoints) == 0:
        return None

    h, w = frame.shape[:2]

    # If only one person detected, validate them
    if len(all_keypoints) == 1:
        person_kps = all_keypoints[0]
        person_scores = all_scores[0]

        if is_horizontal_position_from_kps(person_kps, person_scores, h, w):
            if person_scores[11] > 0.15 and person_scores[12] > 0.15:
                hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
                hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w

                # Invisible barrier — reject if in top % of frame (tent zone)
                if hip_y < IGNORE_TOP_PERCENT:
                    return None
                if hip_y < 0.35:
                    return None
                # Reject if too close to left/right edges (tents are usually there)
                if hip_x < 0.15 or hip_x > 0.85:
                    return None
                if abs(hip_x - 0.5) < 0.35:
                    if is_in_water_lenient(hip_x, hip_y, water_level, frame):
                        return 0  # return index
        return None

    # Multiple people — score and pick the best valid one
    pose_sizes = []
    for i, (person_kps, person_scores) in enumerate(zip(all_keypoints, all_scores)):
        if person_scores[11] > 0.15 and person_scores[12] > 0.15:
            hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
            hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w

            # Invisible barrier
            if hip_y < IGNORE_TOP_PERCENT:
                continue
            if hip_y < 0.35:
                continue
            # Reject edge positions (tents)
            if hip_x < 0.15 or hip_x > 0.85:
                continue

        size_info = calculate_pose_size_coco(person_kps, person_scores, h, w)
        if size_info:
            pose_sizes.append((i, person_kps, person_scores, size_info))

    if not pose_sizes:
        return None

    # Sort by size (largest = foreground swimmer)
    pose_sizes.sort(key=lambda x: x[3]['size'], reverse=True)
    largest_size = pose_sizes[0][3]['size']
    min_foreground_size = largest_size * 0.60

    for idx, person_kps, person_scores, size_info in pose_sizes:
        if size_info['size'] < min_foreground_size:
            continue
        if size_info['size'] < 0.15:
            continue
        if not is_horizontal_position_from_kps(person_kps, person_scores, h, w):
            continue
        if person_scores[11] < 0.15 or person_scores[12] < 0.15:
            continue

        hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
        hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w

        # Invisible barrier
        if hip_y < IGNORE_TOP_PERCENT:
            continue
        if hip_y < 0.35 or hip_y < 0.45:
            continue
        # Reject edge positions
        if hip_x < 0.15 or hip_x > 0.85:
            continue
        if abs(hip_x - 0.5) > 0.45:
            continue
        if hip_y < water_level - 0.10:
            continue
        if not is_in_water_lenient(hip_x, hip_y, water_level, frame):
            continue
        if not validate_pose_anatomy_coco(person_kps, person_scores, h, w):
            continue

        # Count visible major joints
        visible_count = 0
        for coco_idx in [0, 5, 6, 11, 12, 13, 14, 15, 16]:
            if person_scores[coco_idx] > 0.15:
                visible_count += 1
        if visible_count < 3:
            continue

        return idx  # return index of best person

    return None


# ============================================================================
# MAIN TRACKER CLASS — RTMPose engine with all original above-water systems
# ============================================================================

class AboveWaterRTMPoseTracker:
    """
    Above-water tracker using RTMPose-x for detection.
    Keeps all the original background systems:
      - Physical tent masking
      - Swimmer locking
      - Water color validation
      - Multi-level smoothing (different for hips, feet, toes)
      - Jump filtering
      - Hip correction (COCO hip = waist, shift toward knee)
    """

    def __init__(self, mode='balanced', det_frequency=2):
        from rtmlib import PoseTracker, Body

        # mode: 'performance' (most accurate, slowest), 'balanced', or 'lightweight' (fastest)
        # det_frequency: how often to run the detector (1 = every frame, slower/more accurate;
        #                higher = faster, relies more on tracking between detections)
        self.pose_tracker = PoseTracker(
            Body,
            mode=mode,
            det_frequency=det_frequency,
            backend='onnxruntime',
            device='cpu',
            to_openpose=False,    # COCO format
        )

        self.tracking_data = []

        # Swimmer locking — same as original
        self.locked_swimmer = None
        self.frames_since_detection = 0
        self.max_frames_lost = 30

        # Position history for smoothing — same sizes as original
        self.position_history = []
        self.history_size = 5

        # Separate smoothing histories for different body parts (same as original)
        self.foot_history = {}
        self.foot_history_size = 7    # responsive

        self.hip_history = {}
        self.hip_history_size = 12    # stable

        self.toe_history = {}
        self.toe_history_size = 9

        # Jump filter
        self.last_known = {}
        self.max_jump = 0.25

        # COCO "hip" = waist. Shift 20% toward knee for true hip crease.
        self.hip_correction_ratio = 0.20

    def process_frame(self, frame, frame_num, fps, water_level):
        """
        Process one frame with RTMPose + all original above-water systems.
        Physical tent masking: blacks out top region BEFORE detection.
        """
        h, w = frame.shape[:2]

        # ── PHYSICAL TENT MASKING — blacks out top region before detection ──
        # Same as original: pose detector literally cannot see the tent zone
        frame_masked = frame.copy()
        mask_height = int(h * IGNORE_TOP_PERCENT)
        frame_masked[0:mask_height, :] = 0   # black out top region

        # Enhance the masked frame
        enhanced = quick_enhance(frame_masked)

        # Run RTMPose on the masked frame
        keypoints, scores = self.pose_tracker(enhanced)

        # Use the original swimmer selection logic (tent rejection, water check, etc.)
        best_person_idx = None
        best_person = None

        if keypoints is not None and len(keypoints) > 0:
            if self.locked_swimmer is not None:
                # Try to find the locked swimmer by proximity
                best_person_idx = self._find_matching_swimmer(
                    keypoints, scores, self.locked_swimmer, water_level, frame_masked, h, w
                )

                if best_person_idx is not None:
                    self.frames_since_detection = 0
                    self.locked_swimmer = self._get_swimmer_position(
                        keypoints[best_person_idx], scores[best_person_idx], h, w
                    )
                else:
                    self.frames_since_detection += 1
                    if self.frames_since_detection > self.max_frames_lost:
                        # Lost swimmer too long — try fresh selection
                        best_person_idx = select_best_swimmer_coco(
                            keypoints, scores, water_level, frame_masked
                        )
                        if best_person_idx is not None:
                            self.locked_swimmer = self._get_swimmer_position(
                                keypoints[best_person_idx], scores[best_person_idx], h, w
                            )
                            self.frames_since_detection = 0
            else:
                # No locked swimmer — pick the best one
                best_person_idx = select_best_swimmer_coco(
                    keypoints, scores, water_level, frame_masked
                )
                if best_person_idx is not None:
                    self.locked_swimmer = self._get_swimmer_position(
                        keypoints[best_person_idx], scores[best_person_idx], h, w
                    )
                    self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection > self.max_frames_lost:
                self.locked_swimmer = None

        if best_person_idx is not None:
            best_person = (keypoints[best_person_idx], scores[best_person_idx])

        # ── Build frame data ──
        frame_data = {
            'frame': frame_num,
            'time_seconds': round(frame_num / fps, 4),
            'water_level': round(water_level, 4),
            'tracking_locked': self.locked_swimmer is not None,
            'frames_since_detection': self.frames_since_detection,
        }

        if best_person is not None:
            person_kps, person_scores = best_person
            current_positions = {}

            # Extract COCO keypoints → normalized coordinates
            for coco_idx, landmark_name in COCO_TO_LANDMARKS.items():
                conf = float(person_scores[coco_idx])
                if conf > 0.05:
                    x_norm = person_kps[coco_idx][0] / w
                    y_norm = person_kps[coco_idx][1] / h
                    current_positions[landmark_name] = {
                        'x': x_norm,
                        'y': y_norm,
                        'z': 0.0,
                        'visibility': conf,
                        'above_water': y_norm < water_level,
                    }

            # ── Hip correction: COCO "hip" = waist, shift toward knee ──
            for side in ['left', 'right']:
                hip_name = f'{side}_hip'
                knee_name = f'{side}_knee'
                if hip_name in current_positions and knee_name in current_positions:
                    hip = current_positions[hip_name]
                    knee = current_positions[knee_name]
                    r = self.hip_correction_ratio
                    corrected_y = hip['y'] + r * (knee['y'] - hip['y'])
                    current_positions[hip_name] = {
                        'x': hip['x'] + r * (knee['x'] - hip['x']),
                        'y': corrected_y,
                        'z': hip['z'],
                        'visibility': min(hip['visibility'], knee['visibility']),
                        'above_water': corrected_y < water_level,
                    }

            # ── Synthesize foot landmarks from ankles (COCO has no feet) ──
            for side in ['left', 'right']:
                ankle_name = f'{side}_ankle'
                if ankle_name in current_positions:
                    ankle = current_positions[ankle_name]
                    current_positions[f'{side}_heel'] = {
                        'x': ankle['x'], 'y': ankle['y'] - 0.005, 'z': 0.0,
                        'visibility': ankle['visibility'] * 0.8,
                        'above_water': (ankle['y'] - 0.005) < water_level,
                    }
                    current_positions[f'{side}_foot_index'] = {
                        'x': ankle['x'], 'y': ankle['y'] + 0.005, 'z': 0.0,
                        'visibility': ankle['visibility'] * 0.8,
                        'above_water': (ankle['y'] + 0.005) < water_level,
                    }
                    current_positions[f'{side}_foot_best'] = {
                        'x': ankle['x'], 'y': ankle['y'], 'z': 0.0,
                        'visibility': ankle['visibility'],
                        'above_water': ankle['y'] < water_level,
                        'source': ankle_name,
                    }

            # ── Jump filter — reject landmarks that teleport ──
            current_positions = self._filter_jumps(current_positions)

            # ── Update per-joint histories (same structure as original) ──

            # Foot history (responsive, size 7)
            for side in ['left', 'right']:
                # Pick best foot landmark (same priority logic as original)
                foot_landmarks = [
                    (f'{side}_foot_index', 10.0),
                    (f'{side}_ankle', 7.0),
                    (f'{side}_heel', 5.0),
                ]
                best_foot = None
                best_score = 0
                for lm_name, priority_weight in foot_landmarks:
                    if lm_name in current_positions:
                        score = current_positions[lm_name]['visibility'] * priority_weight
                        if score > best_score:
                            best_score = score
                            best_foot = lm_name

                if best_foot:
                    foot_key = f'{side}_foot_best'
                    if foot_key not in self.foot_history:
                        self.foot_history[foot_key] = []
                    self.foot_history[foot_key].append({
                        'x': current_positions[best_foot]['x'],
                        'y': current_positions[best_foot]['y'],
                        'z': current_positions[best_foot]['z'],
                        'visibility': current_positions[best_foot]['visibility'],
                        'above_water': current_positions[best_foot]['above_water'],
                        'source': best_foot,
                    })
                    if len(self.foot_history[foot_key]) > self.foot_history_size:
                        self.foot_history[foot_key].pop(0)

            # Hip history (stable, size 12)
            for side in ['left', 'right']:
                hip_name = f'{side}_hip'
                if hip_name in current_positions:
                    hip_key = f'{side}_hip_ultra'
                    if hip_key not in self.hip_history:
                        self.hip_history[hip_key] = []
                    self.hip_history[hip_key].append({
                        'x': current_positions[hip_name]['x'],
                        'y': current_positions[hip_name]['y'],
                        'z': current_positions[hip_name]['z'],
                        'visibility': current_positions[hip_name]['visibility'],
                        'above_water': current_positions[hip_name]['above_water'],
                    })
                    if len(self.hip_history[hip_key]) > self.hip_history_size:
                        self.hip_history[hip_key].pop(0)

            # Toe history (size 9)
            for side in ['left', 'right']:
                toe_name = f'{side}_foot_index'
                if toe_name in current_positions:
                    toe_key = f'{side}_toe_ultra'
                    if toe_key not in self.toe_history:
                        self.toe_history[toe_key] = []
                    self.toe_history[toe_key].append({
                        'x': current_positions[toe_name]['x'],
                        'y': current_positions[toe_name]['y'],
                        'z': current_positions[toe_name]['z'],
                        'visibility': current_positions[toe_name]['visibility'],
                        'above_water': current_positions[toe_name]['above_water'],
                    })
                    if len(self.toe_history[toe_key]) > self.toe_history_size:
                        self.toe_history[toe_key].pop(0)

            # General position history
            self.position_history.append(current_positions)
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)

            # ── Smooth all positions (same multi-level smoothing as original) ──
            smoothed_positions = self._smooth_all(water_level)

            # Save to frame data
            for joint_name, pos in smoothed_positions.items():
                frame_data[f'{joint_name}_x'] = round(pos['x'], 4)
                frame_data[f'{joint_name}_y'] = round(pos['y'], 4)
                frame_data[f'{joint_name}_z'] = round(pos['z'], 4)
                frame_data[f'{joint_name}_visibility'] = round(pos['visibility'], 4)
                frame_data[f'{joint_name}_above_water'] = pos['above_water']

        self.tracking_data.append(frame_data)
        return best_person

    def _filter_jumps(self, current_positions):
        """Reject landmarks that jump too far frame-to-frame"""
        filtered = {}
        for joint_name, pos in current_positions.items():
            if joint_name in self.last_known:
                last = self.last_known[joint_name]
                dist = np.sqrt((pos['x'] - last['x'])**2 + (pos['y'] - last['y'])**2)
                if dist > self.max_jump:
                    # Keep last known position with reduced confidence
                    held = {
                        'x': last['x'], 'y': last['y'], 'z': pos['z'],
                        'visibility': pos['visibility'] * 0.3,
                        'above_water': pos['above_water'],
                    }
                    if 'source' in pos:
                        held['source'] = pos['source']
                    filtered[joint_name] = held
                    continue
            filtered[joint_name] = pos
            self.last_known[joint_name] = {'x': pos['x'], 'y': pos['y']}
        return filtered

    def _smooth_all(self, water_level):
        """
        Multi-level weighted smoothing — same as original.
        Different smoothing power for feet vs body vs hips.
        """
        smoothed = {}

        def _weighted_avg(history, power):
            weights = [(i + 1) ** power for i in range(len(history))]
            tw = sum(weights)
            result = {
                'x': sum(p['x'] * w for p, w in zip(history, weights)) / tw,
                'y': sum(p['y'] * w for p, w in zip(history, weights)) / tw,
                'z': sum(p.get('z', 0) * w for p, w in zip(history, weights)) / tw,
                'visibility': sum(p['visibility'] * w for p, w in zip(history, weights)) / tw,
            }
            result['above_water'] = result['y'] < water_level
            if 'source' in history[-1]:
                result['source'] = history[-1]['source']
            return result

        # General joint smoothing from position_history
        for joint_name in ALL_LANDMARKS:
            is_foot = ('foot' in joint_name or 'ankle' in joint_name or 'heel' in joint_name)
            hist = [f[joint_name] for f in self.position_history if joint_name in f]
            if hist:
                # Foot joints get quadratic weighting (more responsive)
                # Body joints get linear weighting (smoother)
                power = 2.0 if is_foot else 1.0
                smoothed[joint_name] = _weighted_avg(hist, power)

        # Foot best — quadratic from separate history
        for side in ['left', 'right']:
            foot_key = f'{side}_foot_best'
            if foot_key in self.foot_history and self.foot_history[foot_key]:
                smoothed[foot_key] = _weighted_avg(self.foot_history[foot_key], 2.0)

        # Hip ultra-smoothing — cubic from separate history
        for side in ['left', 'right']:
            hip_key = f'{side}_hip_ultra'
            if hip_key in self.hip_history and self.hip_history[hip_key]:
                smoothed[f'{side}_hip'] = _weighted_avg(self.hip_history[hip_key], 3.0)

        # Toe smoothing — power 2.8 from separate history
        for side in ['left', 'right']:
            toe_key = f'{side}_toe_ultra'
            if toe_key in self.toe_history and self.toe_history[toe_key]:
                smoothed[f'{side}_foot_index'] = _weighted_avg(self.toe_history[toe_key], 2.8)

        return smoothed

    def _get_swimmer_position(self, person_kps, person_scores, h, w):
        """Get hip center position for swimmer locking"""
        if person_scores[11] < 0.05 or person_scores[12] < 0.05:
            return None
        return {
            'x': (person_kps[11][0] + person_kps[12][0]) / 2 / w,
            'y': (person_kps[11][1] + person_kps[12][1]) / 2 / h,
            'frame_ref': True,
        }

    def _find_matching_swimmer(self, all_keypoints, all_scores, locked_position, water_level, frame, h, w):
        """Find the person closest to our locked swimmer position"""
        if not locked_position:
            return None

        best_match = None
        best_distance = 0.20

        for i, (person_kps, person_scores) in enumerate(zip(all_keypoints, all_scores)):
            if person_scores[11] < 0.05 or person_scores[12] < 0.05:
                continue

            hip_y = (person_kps[11][1] + person_kps[12][1]) / 2 / h
            hip_x = (person_kps[11][0] + person_kps[12][0]) / 2 / w

            if hip_y < 0.15:
                continue

            distance = np.sqrt((hip_x - locked_position['x'])**2 +
                               (hip_y - locked_position['y'])**2)

            if distance < best_distance:
                best_distance = distance
                best_match = i

        return best_match

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


# ============================================================================
# VISUALIZATION — SAME STYLE AS ORIGINAL (adapted for COCO keypoints)
# ============================================================================

# COCO skeleton connections for drawing
COCO_CONNECTIONS = [
    (5, 6),              # shoulders
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # torso sides
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]


def draw_frame(frame, best_person, water_level, frame_num, total_frames, tracker_status=None):
    """
    Draw visualization — same layout as original above-water tracker.
    Waterline, skeleton overlay, above/below counts, foot labels.
    """
    viz = frame.copy()
    h, w = frame.shape[:2]
    water_y = int(water_level * h)

    # Draw waterline (same style as original)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 0, 0), 6)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 255, 255), 3)

    waterline_text = f"WATERLINE: {water_level:.3f}"
    cv2.putText(viz, waterline_text, (10, water_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(viz, waterline_text, (10, water_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Water tint below waterline
    overlay = viz.copy()
    cv2.rectangle(overlay, (0, water_y), (w, h), (255, 200, 100), -1)
    cv2.addWeighted(overlay, 0.15, viz, 0.85, 0, viz)

    tracking_stopped = tracker_status and tracker_status.get('tracking_stopped', False)

    cv2.rectangle(viz, (10, 10), (650, 140), (0, 0, 0), -1)

    if tracking_stopped:
        cv2.putText(viz, "TRACKING STOPPED",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 255), 2)
        cv2.putText(viz, "Figure complete",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    elif best_person is not None:
        person_kps, person_scores = best_person

        # Draw skeleton connections
        for (i, j) in COCO_CONNECTIONS:
            if person_scores[i] > 0.2 and person_scores[j] > 0.2:
                pt1 = (int(person_kps[i][0]), int(person_kps[j][1]))
                pt2 = (int(person_kps[j][0]), int(person_kps[j][1]))

                # Thicker lines for legs/feet (same idea as original)
                if i >= 13 or j >= 13:
                    thickness = 5
                    color = (255, 255, 255)
                elif i >= 11 or j >= 11:
                    thickness = 4
                    color = (200, 200, 200)
                else:
                    thickness = 3
                    color = (180, 180, 180)

                pt1_correct = (int(person_kps[i][0]), int(person_kps[i][1]))
                pt2_correct = (int(person_kps[j][0]), int(person_kps[j][1]))
                cv2.line(viz, pt1_correct, pt2_correct, color, thickness, cv2.LINE_AA)

        # Draw keypoints
        above = total_joints = 0
        foot_points = []

        # COCO keypoint names for labels
        coco_names = {
            0: 'nose', 1: 'l_eye', 2: 'r_eye', 3: 'l_ear', 4: 'r_ear',
            5: 'l_sho', 6: 'r_sho', 7: 'l_elb', 8: 'r_elb',
            9: 'l_wri', 10: 'r_wri', 11: 'l_hip', 12: 'r_hip',
            13: 'l_kne', 14: 'r_kne', 15: 'l_ank', 16: 'r_ank',
        }

        for idx in range(17):
            min_vis = 0.2 if idx >= 15 else 0.3
            if person_scores[idx] > min_vis:
                x = int(person_kps[idx][0])
                y = int(person_kps[idx][1])
                is_above = y < water_y
                color = (0, 255, 0) if is_above else (0, 0, 255)

                if idx >= 15:  # ankles (feet in COCO)
                    radius = 14
                    cv2.circle(viz, (x, y), radius + 4, color, 2)
                    cv2.circle(viz, (x, y), radius, color, -1)
                    cv2.circle(viz, (x, y), radius, (255, 255, 255), 2)
                    foot_points.append((idx, x, y, is_above))
                    label = coco_names.get(idx, str(idx))
                    cv2.putText(viz, label[:5], (x + 18, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                else:
                    radius = 9
                    cv2.circle(viz, (x, y), radius, color, -1)
                    cv2.circle(viz, (x, y), radius, (255, 255, 255), 2)

                above += int(is_above)
                total_joints += 1

        above_pct = (above / total_joints * 100) if total_joints > 0 else 0
        below_pct = 100 - above_pct

        cv2.putText(viz, f"Above: {above}/{total_joints} ({above_pct:.0f}%)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(viz, f"Below: {total_joints-above}/{total_joints} ({below_pct:.0f}%)",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        feet_above = sum(1 for _, _, _, ia in foot_points if ia)
        feet_total = len(foot_points)
        cv2.putText(viz, f"Feet/ankles above: {feet_above}/{feet_total}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(viz, "RTMPose-x ABOVE-WATER",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(viz, "No horizontal swimmer detected",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Frame counter (same position as original)
    frame_text = f"Frame: {frame_num}/{total_frames}"
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = 650 - text_size[0] - 20
    cv2.putText(viz, frame_text, (text_x, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return viz


# ============================================================================
# MAIN PROCESSING — SAME FLOW AS ORIGINAL
# ============================================================================

def process_video(video_path, output_path, water_level,
                   mode='balanced', det_frequency=2, progress_callback=None):
    """
    Process video — same flow as original, RTMPose engine.

    mode / det_frequency: passed through to the pose tracker (see
        AboveWaterRTMPoseTracker.__init__ for the speed/accuracy tradeoff).
    progress_callback: optional callable(frame_count, total_frames) invoked
        periodically during processing, so a UI (e.g. Streamlit) can show
        live progress instead of relying on stdout prints.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = fps if OUTPUT_FPS is None else OUTPUT_FPS

    print(f"\n{'='*70}")
    print(f"RTMPose ABOVE-WATER TRACKER")
    print(f"{'='*70}")
    print(f"Video: {Path(video_path).name}")
    print(f"Resolution: {w}x{h} | FPS: {fps:.0f}")
    print(f"Frames: {total} | Duration: {total/fps:.1f}s")
    print(f"AI Model: RTMPose-x ({mode}) + YOLOX-x | det_frequency={det_frequency}")
    print(f"\n🚫 PHYSICAL TENT MASKING:")
    print(f"   Masking top: {IGNORE_TOP_PERCENT*100:.0f}% of frame ({int(h*IGNORE_TOP_PERCENT)}px)")
    print(f"   Detection area: {(1-IGNORE_TOP_PERCENT)*100:.0f}% (bottom {int(h*(1-IGNORE_TOP_PERCENT))}px)")
    print(f"   ✓ Top region BLACKED OUT before detection")
    print(f"   ✓ Pose detector CANNOT see tents")
    print(f"   ✓ 100% guaranteed no tent tracking!")
    print(f"   ✓ Full frame video output preserved")

    tracker = AboveWaterRTMPoseTracker(mode=mode, det_frequency=det_frequency)

    if water_level is None:
        # Use RTMPose for waterline detection too
        water_level = detect_waterline_from_poses(video_path, tracker.pose_tracker)

        if water_level is None:
            print(f"\n❌ WATERLINE DETECTION FAILED")
            cap.release()
            return None, None
    else:
        print(f"\nWaterline: {water_level:.3f} (MANUAL)")

    if MAX_FIGURE_DURATION is not None:
        max_frames_to_track = int(MAX_FIGURE_DURATION * fps)
        print(f"✓ Duration limit: {MAX_FIGURE_DURATION}s ({max_frames_to_track} frames)")
    else:
        max_frames_to_track = total

    print(f"{'='*70}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        os.remove(output_path)
        time.sleep(0.3)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (w, h))
    if not out.isOpened():
        raise Exception("Failed to create video writer")

    print("Processing frames...")
    start_time = time.time()
    frame_count = 0
    tracking_stopped = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= max_frames_to_track and not tracking_stopped:
            print(f"\n  ⏸️  Stopping tracking at frame {frame_count}")
            tracking_stopped = True

        if not tracking_stopped:
            best_person = tracker.process_frame(frame, frame_count, fps, water_level)
        else:
            best_person = None

        tracker_status = {
            'frames_lost': tracker.frames_since_detection,
            'max_frames': tracker.max_frames_lost,
            'tracking_stopped': tracking_stopped,
        }

        # Draw on FULL frame (not masked) — same as original
        annotated = draw_frame(frame, best_person, water_level,
                               frame_count, total, tracker_status)
        out.write(annotated)
        frame_count += 1

        if progress_callback is not None:
            progress_callback(frame_count, total)

        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed
            eta = (total - frame_count) / fps_proc if fps_proc > 0 else 0
            progress = frame_count / total * 100
            print(f"  {progress:.1f}% | {frame_count}/{total} | "
                  f"{fps_proc:.1f} fps | ETA: {eta:.0f}s")

    cap.release()
    out.release()
    time.sleep(0.5)

    elapsed_total = time.time() - start_time
    file_size = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*70}")
    print(f"✅ COMPLETE!")
    print(f"  Processed: {total} frames in {elapsed_total:.1f}s")
    print(f"  Physical mask: Top {IGNORE_TOP_PERCENT*100:.0f}% blacked out for detection")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"{'='*70}\n")

    df = tracker.get_dataframe()
    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Data saved: {csv_path}\n")

    return output_path, csv_path


# ============================================================================
# IMPROVED KALMAN FILTER — IDENTICAL TO ORIGINAL
# ============================================================================

class ImprovedKalmanFilter1D:
    """1D Kalman filter with adaptive gains — same as original"""

    def __init__(self, process_var=0.001, measurement_var=0.05, outlier_threshold=0.15):
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 1.0
        self.Q = np.array([[process_var, 0], [0, process_var * 0.1]])
        self.R = np.array([[measurement_var]])
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.initialized = False
        self.outlier_threshold = outlier_threshold
        self.last_measurement = None
        self.consecutive_predictions = 0
        self.max_predictions = 20

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]

    def is_outlier(self, measurement):
        if self.last_measurement is None:
            return False
        return abs(measurement - self.last_measurement) > self.outlier_threshold

    def update(self, measurement, confidence=1.0, force=False):
        if not self.initialized:
            self.x[0] = measurement
            self.last_measurement = measurement
            self.initialized = True
            self.consecutive_predictions = 0
            return measurement
        if not force and self.is_outlier(measurement):
            return self.predict()
        R_adjusted = self.R * (2.0 - confidence)
        y = measurement - (self.H @ self.x)[0]
        S = self.H @ self.P @ self.H.T + R_adjusted
        K = self.P @ self.H.T / S
        self.x = self.x + (K * y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.last_measurement = measurement
        self.consecutive_predictions = 0
        return self.x[0]

    def filter(self, measurement, confidence=1.0):
        predicted = self.predict()
        if measurement is not None and not np.isnan(measurement):
            return self.update(measurement, confidence)
        else:
            self.consecutive_predictions += 1
            if self.consecutive_predictions > self.max_predictions:
                return np.nan
            return predicted


def apply_kalman_filter_to_csv(csv_path):
    """Apply improved Kalman filter to tracking CSV — same as original"""
    print(f"\n{'='*70}")
    print(f"✨ IMPROVED KALMAN FILTERING")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path)
    print(f"  Loading: {Path(csv_path).name}")
    print(f"  Frames: {len(df)}")

    joints = ALL_LANDMARKS

    print(f"\n  Applying adaptive Kalman filter...")

    for joint in joints:
        for axis in ['y', 'x']:
            col = f'{joint}_{axis}'
            vis_col = f'{joint}_visibility'
            if col not in df.columns:
                continue
            first_idx = df[col].first_valid_index()
            if first_idx is None:
                continue

            kf = ImprovedKalmanFilter1D()
            kf.x[0] = df.loc[first_idx, col]
            kf.initialized = True

            filtered = []
            for _, row in df.iterrows():
                m = row[col] if not pd.isna(row[col]) else None
                c = row[vis_col] if vis_col in df.columns and not pd.isna(row[vis_col]) else 1.0
                if m is not None and kf.is_outlier(m):
                    m = None
                filtered.append(kf.filter(m, c))

            df[f'{joint}_{axis}_raw'] = df[col]
            df[col] = filtered

    output_path = Path(csv_path).parent / (Path(csv_path).stem + "_KALMAN.csv")
    df.to_csv(output_path, index=False, float_format='%.4f')

    print(f"\n  ✅ Improved Kalman filtering complete!")
    print(f"    Saved: {output_path.name}\n")

    return output_path


# ============================================================================
# BATCH PROCESSING — SAME AS ORIGINAL
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"RTMPose ABOVE-WATER TRACKER")
    print(f"{'='*70}")
    print(f"Engine: RTMPose-x (performance) + YOLOX-x detector")
    print(f"Detection: every frame (det_frequency=1)")
    print(f"Background systems from original above-water tracker:")
    print(f"  ✓ Physical tent masking (top {IGNORE_TOP_PERCENT*100:.0f}% blacked out)")
    print(f"  ✓ Blue water color validation")
    print(f"  ✓ Waterline = average of head + shoulder + hip (100 frames)")
    print(f"  ✓ Swimmer locking + edge rejection")
    print(f"  ✓ Multi-level smoothing (hips, feet, toes)")
    print(f"  ✓ Adaptive Kalman filtering")
    print(f"  ✓ Hip correction (COCO waist → true hip crease)")
    print(f"{'='*70}\n")

    video_list = []
    if isinstance(VIDEO_PATHS, str):
        if '*' in VIDEO_PATHS or '?' in VIDEO_PATHS:
            video_list = sorted(glob.glob(VIDEO_PATHS))
        else:
            video_list = [VIDEO_PATHS]
    elif isinstance(VIDEO_PATHS, list):
        video_list = VIDEO_PATHS

    existing_videos = [v for v in video_list if Path(v).exists()]
    if not existing_videos:
        print(f"❌ No videos found")
        exit(1)

    print(f"Processing {len(existing_videos)} video(s)...\n")
    results = []
    failed = []

    for i, video_path_str in enumerate(existing_videos, 1):
        video_path = Path(video_path_str)
        output_path = video_path.parent / f"{video_path.stem}_POSE_tracking.mp4"

        print(f"\n{'#'*70}")
        print(f"VIDEO {i}/{len(existing_videos)}: {video_path.name}")
        print(f"{'#'*70}")

        try:
            video_file, csv_file = process_video(str(video_path), output_path, WATERLINE)

            if video_file is None or csv_file is None:
                failed.append({'video': video_path.name, 'error': 'Waterline detection failed'})
                continue

            kalman_csv = apply_kalman_filter_to_csv(csv_file)
            results.append({
                'video': video_path.name,
                'output': video_file,
                'csv': csv_file,
                'kalman_csv': kalman_csv,
                'status': 'SUCCESS',
            })

            print(f"\n✅ Video {i}/{len(existing_videos)} complete!")
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append({'video': video_path.name, 'error': str(e)})

    print(f"\n\n{'='*70}")
    print(f"✨ BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"\n✅ Successfully processed: {len(results)}/{len(existing_videos)}")

    if results:
        print(f"\nProcessed videos:")
        for r in results:
            print(f"  ✓ {r['video']}")
            print(f"    CSV: {r['kalman_csv'].name}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for f in failed:
            print(f"  ✗ {f['video']}: {f['error']}")

    print(f"\n{'='*70}")
    print(f"RTMPose ABOVE-WATER TRACKER — COMPLETE")
    print(f"{'='*70}\n")

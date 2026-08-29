#!/usr/bin/env python3
"""
UNDERWATER TRACKER v4 — RTMPose MAX ACCURACY + VALIDATED LOCKING
===========================================================================
Uses RTMPose-x (via rtmlib) with maximum accuracy settings.

============================================================================
v4 CHANGES — bringing this up to parity with the above-water tracker and
the WaltiCam v2 tracker, which both already have this; this file did not.
============================================================================
v3 picked "whichever detected person has the highest confidence+size"
FRESH, EVERY SINGLE FRAME, with no persistence and no anatomy validation.
That's the exact bug pattern already found and fixed twice elsewhere in
this project (the above-water tracker's original _find_matching_swimmer,
and the WaltiCam v1 tracker before it got SwimmerLock) — a second swimmer
in an adjacent lane, a coach standing poolside and visible through the
water, or a momentary bad detection could silently steal the identity
frame-to-frame, with nothing to catch it.

v4 adds:
  1. Real locking. Once a swimmer is locked, later frames must match by
     proximity to the last known position AND pass anatomy validation —
     not just "whoever scores highest this frame."
  2. Anatomy validation (validate_pose_anatomy_coco) — shoulder-above-hip
     sanity, hip-width sanity. Pure geometry, same check used in the
     above-water tracker.
  3. Periodic re-validation every RELOCK_CHECK_INTERVAL frames, requiring
     TWO consecutive disagreements before switching who's tracked — a
     one-off outlier can't hijack a good lock, but a sustained one still
     gets corrected.
  4. Capped hold-through for brief detection gaps (SHORT_GAP_HOLD_FRAMES),
     instead of holding the last position indefinitely with no limit —
     v3's `elif self.position_history:` held forever with no cap, which
     could quietly paper over a real, sustained loss as if tracking were
     still working.

v4 deliberately does NOT copy the above-water tracker's water-color
check — the whole frame is already underwater here, so there's no
tent/deck/background-person-standing-in-air scenario to distinguish
against by color the way there is above water. Position/anatomy
validation covers the actual risk (an adjacent swimmer, a coach visible
through the water) without adding a color-calibration dependency that
isn't needed for this context.

v4 keeps everything that was already good and underwater-specific,
UNCHANGED:
  - mode='performance', det_frequency=1 (max accuracy)
  - Looser Kalman filter (process_var=0.005, measurement_var=0.03,
    outlier_threshold=0.20) — deliberately looser than above-water's,
    tuned for underwater's faster/more erratic thrust movement
  - Shorter position history (less lag) and hip-history smoothing
  - The synthesized mid_spine keypoint for back-curvature measurements
  - Waterline detection (edge + brightness-gradient + shoulder
    calibration) — already reasonably robust, not exhibiting the same
    bug pattern the person-selection logic had

INSTALL:
    pip install rtmlib onnxruntime opencv-python numpy pandas scipy

OUTPUT: Same CSV format as above-water tracker
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import time
import glob
from scipy import signal

# ============================================================================
# CONFIGURATION
# ============================================================================

VIDEO_PATHS = [
   "/Users/mona/Desktop/Science fairs/Science fair 2026/Barracuda folders/Jmeet figures.nosync/swimmer_21_below.mp4"
]

OUTPUT_FPS = None
WATERLINE = None
MAX_FIGURE_DURATION = 60

# Locking / re-validation tuning (v4) — same values as the above-water
# tracker, since the underlying logic and risk profile are the same.
DET_FREQUENCY = 1
LOCK_MATCH_DISTANCE = 0.15
RELOCK_CHECK_INTERVAL = 45      # ~1.5s at 30fps
RELOCK_DRIFT_THRESHOLD = 0.15
MAX_FRAMES_LOST = 30
SHORT_GAP_HOLD_FRAMES = 5       # hold last known position through brief gaps only
EDGE_MARGIN = 0.05              # reject hips right at the very edge of frame

# RTMPose COCO keypoint indices → our landmark names
COCO_TO_LANDMARKS = {
    0: 'nose',
    5: 'left_shoulder',
    6: 'right_shoulder',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}

# All landmark names we output (matching MediaPipe tracker format)
ALL_LANDMARKS = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
    'left_foot_best', 'right_foot_best',
    'mid_spine',
]

FOOT_LANDMARKS = ['left_ankle', 'right_ankle', 'left_heel', 'right_heel',
                  'left_foot_index', 'right_foot_index']


# ============================================================================
# UNDERWATER IMAGE ENHANCEMENT
# ============================================================================

def enhance_underwater(frame):
    """Underwater color correction + CLAHE + light sharpen"""
    b, g, r = cv2.split(frame)
    r_boosted  = cv2.convertScaleAbs(r, alpha=1.4, beta=10)
    g_adjusted = cv2.convertScaleAbs(g, alpha=1.1, beta=0)
    b_reduced  = cv2.convertScaleAbs(b, alpha=0.85, beta=0)
    corrected = cv2.merge([b_reduced, g_adjusted, r_boosted])

    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a, b_ch]), cv2.COLOR_LAB2BGR)

    kernel = np.array([[-0.5, -0.5, -0.5],
                       [-0.5,  5.0, -0.5],
                       [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return cv2.addWeighted(sharpened, 0.6, enhanced, 0.4, 0)


# ============================================================================
# ANATOMICAL VALIDATION (new in v4 — same geometry check used by the
# above-water tracker; pure proportions, no color/venue dependency)
# ============================================================================

def validate_pose_anatomy_coco(person_kps, person_scores, h, w):
    """Reject poses with implausible proportions (e.g. a partial/merged
    detection, or two overlapping swimmers read as one)."""
    if person_scores[5] > 0.10 and person_scores[11] > 0.10:
        ls_y = person_kps[5][1] / h
        lh_y = person_kps[11][1] / h
        if ls_y > lh_y + 0.35:
            return False
    if person_scores[6] > 0.10 and person_scores[12] > 0.10:
        rs_y = person_kps[6][1] / h
        rh_y = person_kps[12][1] / h
        if rs_y > rh_y + 0.35:
            return False
    if person_scores[11] > 0.10 and person_scores[12] > 0.10:
        lh_x = person_kps[11][0] / w
        rh_x = person_kps[12][0] / w
        hip_width = abs(lh_x - rh_x)
        if hip_width < 0.02 or hip_width > 0.60:
            return False
    return True


def get_hip_position(person_kps, person_scores, h, w):
    if person_scores[11] < 0.05 or person_scores[12] < 0.05:
        return None
    return {
        'x': (person_kps[11][0] + person_kps[12][0]) / 2 / w,
        'y': (person_kps[11][1] + person_kps[12][1]) / 2 / h,
    }


# ============================================================================
# WATERLINE DETECTION (unchanged from v3)
# ============================================================================

def detect_underwater_waterline(video_path):
    """Detect waterline from underwater footage"""
    print("\nUnderwater waterline detection...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    search_frames = min(200, total_frames)
    edge_candidates = []
    color_candidates = []

    for frame_num in range(0, search_frames, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        search_end = int(h * 0.50)
        search_region = edges[0:search_end, :]
        horizontal_sum = np.sum(search_region, axis=1)
        if len(horizontal_sum) > 0 and np.max(horizontal_sum) > w * 0.20:
            edge_candidates.append(np.argmax(horizontal_sum) / h)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]
        avg_brightness = np.mean(value_channel[0:search_end, :], axis=1)
        if len(avg_brightness) > 10:
            win = min(11, len(avg_brightness) // 2 * 2 + 1)
            if win >= 5:
                smoothed = signal.savgol_filter(avg_brightness, window_length=win, polyorder=2)
                gradient = np.diff(smoothed)
                if len(gradient) > 0:
                    max_idx = np.argmax(gradient)
                    if gradient[max_idx] > 10:
                        color_candidates.append(max_idx / h)

    cap.release()

    all_c = []
    for cands in [edge_candidates, color_candidates]:
        if len(cands) >= 8:
            med = np.median(cands)
            std = np.std(cands)
            all_c.extend([c for c in cands if abs(c - med) < 2 * std])

    if len(all_c) >= 10:
        waterline = float(np.median(all_c))
        if waterline > 0.50:
            waterline = 0.25
    else:
        waterline = 0.25

    print(f"  Waterline: {waterline:.3f}")
    return waterline


def calibrate_waterline_to_shoulders(video_path, initial_waterline, pose_tracker):
    """Set waterline at shoulder level from first 100 frames.
    Uses median (robust to outliers) and requires high confidence."""
    print("  Calibrating waterline to shoulder level (first 100 frames)...")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shoulder_ys = []

    for frame_num in range(0, min(100, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        enhanced = enhance_underwater(frame)
        keypoints, scores = pose_tracker(enhanced)

        if keypoints is None or len(keypoints) == 0:
            continue

        for person_kps, person_scores in zip(keypoints, scores):
            ls_y = person_kps[5][1] / h
            rs_y = person_kps[6][1] / h
            ls_s = person_scores[5]
            rs_s = person_scores[6]

            if ls_s < 0.5 or rs_s < 0.5:
                continue

            shoulder_y = (ls_y + rs_y) / 2

            if 0.05 < shoulder_y < 0.50:
                shoulder_ys.append(shoulder_y)

    cap.release()

    if len(shoulder_ys) >= 5:
        calibrated = float(np.median(shoulder_ys))
        print(f"    {len(shoulder_ys)} samples -> median shoulder: {calibrated:.3f}")
        return calibrated
    else:
        print(f"    Only {len(shoulder_ys)} samples -- keeping {initial_waterline:.3f}")
        return initial_waterline


# ============================================================================
# RTMPose TRACKER
# ============================================================================

class RTMPoseUnderwaterTracker:
    """
    Underwater tracker using RTMPose-x via rtmlib, with validated locking
    (v4) instead of picking the highest-confidence person fresh every
    frame (v3).
    """

    def __init__(self):
        from rtmlib import PoseTracker, Body

        self.pose_tracker = PoseTracker(
            Body,
            mode='performance',       # highest accuracy
            det_frequency=DET_FREQUENCY,
            backend='onnxruntime',
            device='cpu',
            to_openpose=False,        # COCO format
        )

        self.tracking_data = []
        self.position_history = []
        self.history_size = 4         # shorter history = less lag

        self.hip_history = {'left': [], 'right': []}
        self.hip_history_size = 6     # was 14, way too much lag

        self.last_known = {}
        self.max_jump = 0.25          # loose enough for thrust movement

        # v4 — real locking state (was: no locking at all in v3)
        self.locked_swimmer = None
        self.frames_since_detection = 0
        self.max_frames_lost = MAX_FRAMES_LOST
        self.relock_check_interval = RELOCK_CHECK_INTERVAL
        self.relock_drift_threshold = RELOCK_DRIFT_THRESHOLD
        self._frames_since_relock_check = 0
        self._pending_relock_idx = None
        self._pending_relock_streak = 0

        # COCO "hip" keypoints are actually at the waist.
        # Shift them 20% toward the knee to approximate the true hip crease.
        self.hip_correction_ratio = 0.20

    # ── v4: selection helpers ──────────────────────────────────────────

    def _select_best_underwater(self, keypoints, scores, h, w):
        """Fresh (unlocked) selection: confidence+size, validated by
        anatomy and basic edge rejection — this is what v3 did EVERY
        frame; v4 only calls it when there's no lock, or the lock has
        been genuinely missing for max_frames_lost frames."""
        best_idx, best_score = None, -1
        for i, (kps, sc) in enumerate(zip(keypoints, scores)):
            if not validate_pose_anatomy_coco(kps, sc, h, w):
                continue
            hip = get_hip_position(kps, sc, h, w)
            if hip is None:
                continue
            if hip['x'] < EDGE_MARGIN or hip['x'] > 1 - EDGE_MARGIN:
                continue
            major_indices = [5, 6, 11, 12, 13, 14, 15, 16]
            avg_conf = np.mean([sc[j] for j in major_indices])
            hip_width = abs(kps[11][0] - kps[12][0]) / w
            total_score = avg_conf + hip_width * 2.0
            if total_score > best_score:
                best_score, best_idx = total_score, i
        return best_idx

    def _find_matching_underwater(self, keypoints, scores, locked_position, h, w):
        """Locked matching: closest hip to last known position, among
        candidates that pass anatomy validation and aren't right at the
        frame edge (an adjacent-lane swimmer partially in frame)."""
        best_match, best_distance = None, LOCK_MATCH_DISTANCE
        for i, (kps, sc) in enumerate(zip(keypoints, scores)):
            if sc[11] < 0.10 or sc[12] < 0.10:
                continue
            if not validate_pose_anatomy_coco(kps, sc, h, w):
                continue
            hip_x = (kps[11][0] + kps[12][0]) / 2 / w
            hip_y = (kps[11][1] + kps[12][1]) / 2 / h
            if hip_x < EDGE_MARGIN or hip_x > 1 - EDGE_MARGIN:
                continue
            distance = np.sqrt((hip_x - locked_position['x']) ** 2 + (hip_y - locked_position['y']) ** 2)
            if distance < best_distance:
                best_distance, best_match = distance, i
        return best_match

    def process_frame(self, frame, frame_num, fps, water_level):
        """Process a single frame with RTMPose"""
        h, w = frame.shape[:2]

        enhanced = enhance_underwater(frame)
        keypoints, scores = self.pose_tracker(enhanced)

        frame_data = {
            'frame': frame_num,
            'time_seconds': round(frame_num / fps, 4),
            'water_level': round(water_level, 4),
            'tracking_locked': False,
            'frames_since_detection': self.frames_since_detection,
        }

        # v4 — locking state machine (was: fresh best-of-frame selection
        # every single frame in v3, see _select_best_underwater docstring)
        best_idx = None
        if keypoints is not None and len(keypoints) > 0:
            if self.locked_swimmer is not None:
                best_idx = self._find_matching_underwater(keypoints, scores, self.locked_swimmer, h, w)
                if best_idx is not None:
                    self.frames_since_detection = 0

                    # Periodic re-check: require TWO consecutive
                    # disagreements before switching, so a one-off
                    # outlier can't hijack a good lock.
                    self._frames_since_relock_check += 1
                    if self._frames_since_relock_check >= self.relock_check_interval:
                        self._frames_since_relock_check = 0
                        fresh_idx = self._select_best_underwater(keypoints, scores, h, w)
                        if fresh_idx is not None and fresh_idx != best_idx:
                            fresh_pos = get_hip_position(keypoints[fresh_idx], scores[fresh_idx], h, w)
                            cur_pos = get_hip_position(keypoints[best_idx], scores[best_idx], h, w)
                            if fresh_pos is not None and cur_pos is not None:
                                drift = np.sqrt((fresh_pos['x'] - cur_pos['x']) ** 2 + (fresh_pos['y'] - cur_pos['y']) ** 2)
                                if drift > self.relock_drift_threshold:
                                    if self._pending_relock_idx == fresh_idx:
                                        self._pending_relock_streak += 1
                                    else:
                                        self._pending_relock_idx, self._pending_relock_streak = fresh_idx, 1
                                    if self._pending_relock_streak >= 2:
                                        best_idx = fresh_idx
                                        self._pending_relock_idx, self._pending_relock_streak = None, 0
                                else:
                                    self._pending_relock_idx, self._pending_relock_streak = None, 0
                else:
                    self.frames_since_detection += 1
                    if self.frames_since_detection > self.max_frames_lost:
                        best_idx = self._select_best_underwater(keypoints, scores, h, w)
                        if best_idx is not None:
                            self.frames_since_detection = 0
            else:
                best_idx = self._select_best_underwater(keypoints, scores, h, w)
                if best_idx is not None:
                    self.frames_since_detection = 0

            if best_idx is not None:
                self.locked_swimmer = get_hip_position(keypoints[best_idx], scores[best_idx], h, w)
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection > self.max_frames_lost:
                self.locked_swimmer = None

        best_person = (keypoints[best_idx], scores[best_idx]) if best_idx is not None else None
        frame_data['tracking_locked'] = self.locked_swimmer is not None

        if best_person is not None:
            person_kps, person_scores = best_person
            current_positions = {}

            # Extract COCO keypoints → our format
            for coco_idx, landmark_name in COCO_TO_LANDMARKS.items():
                x_px = person_kps[coco_idx][0]
                y_px = person_kps[coco_idx][1]
                conf = float(person_scores[coco_idx])

                if conf > 0.1:
                    x_norm = x_px / w
                    y_norm = y_px / h

                    current_positions[landmark_name] = {
                        'x': x_norm,
                        'y': y_norm,
                        'z': 0.0,
                        'visibility': conf,
                        'above_water': y_norm < water_level,
                    }

            # ── Hip correction: COCO "hip" = waist, shift toward knee ────
            for side in ['left', 'right']:
                hip_name = f'{side}_hip'
                knee_name = f'{side}_knee'
                if hip_name in current_positions and knee_name in current_positions:
                    hip = current_positions[hip_name]
                    knee = current_positions[knee_name]
                    r = self.hip_correction_ratio
                    current_positions[hip_name] = {
                        'x': hip['x'] + r * (knee['x'] - hip['x']),
                        'y': hip['y'] + r * (knee['y'] - hip['y']),
                        'z': hip['z'],
                        'visibility': min(hip['visibility'], knee['visibility']),
                        'above_water': (hip['y'] + r * (knee['y'] - hip['y'])) < water_level,
                    }

            # ── Mid-spine synthesis ──────────────────────────────────────
            # COCO/RTMPose has no keypoint anywhere along the spine — only
            # nose, shoulders, hips, knees, ankles. This adds a mid_spine
            # point as the midpoint between the shoulder-center and the
            # (already hip-corrected) hip-center, giving two segments
            # (hip->mid_spine, mid_spine->shoulder) instead of one, so back
            # curvature becomes visible as an angle rather than invisible.
            if all(k in current_positions for k in
                   ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
                ls = current_positions['left_shoulder']
                rs = current_positions['right_shoulder']
                lh = current_positions['left_hip']
                rh = current_positions['right_hip']

                shoulder_center_x = (ls['x'] + rs['x']) / 2
                shoulder_center_y = (ls['y'] + rs['y']) / 2
                hip_center_x = (lh['x'] + rh['x']) / 2
                hip_center_y = (lh['y'] + rh['y']) / 2

                mid_spine_x = (shoulder_center_x + hip_center_x) / 2
                mid_spine_y = (shoulder_center_y + hip_center_y) / 2
                mid_spine_vis = min(ls['visibility'], rs['visibility'],
                                     lh['visibility'], rh['visibility'])

                current_positions['mid_spine'] = {
                    'x': mid_spine_x,
                    'y': mid_spine_y,
                    'z': 0.0,
                    'visibility': mid_spine_vis,
                    'above_water': mid_spine_y < water_level,
                }

            # Synthesize foot landmarks from ankles (COCO doesn't have feet)
            for side in ['left', 'right']:
                ankle_name = f'{side}_ankle'
                if ankle_name in current_positions:
                    ankle = current_positions[ankle_name]
                    current_positions[f'{side}_heel'] = {
                        'x': ankle['x'],
                        'y': ankle['y'] - 0.005,
                        'z': 0.0,
                        'visibility': ankle['visibility'] * 0.8,
                        'above_water': (ankle['y'] - 0.005) < water_level,
                    }
                    current_positions[f'{side}_foot_index'] = {
                        'x': ankle['x'],
                        'y': ankle['y'] + 0.005,
                        'z': 0.0,
                        'visibility': ankle['visibility'] * 0.8,
                        'above_water': (ankle['y'] + 0.005) < water_level,
                    }
                    current_positions[f'{side}_foot_best'] = {
                        'x': ankle['x'],
                        'y': ankle['y'],
                        'z': 0.0,
                        'visibility': ankle['visibility'],
                        'above_water': ankle['y'] < water_level,
                        'source': ankle_name,
                    }

            # Jump filter
            current_positions = self._filter_jumps(current_positions)

            # Hip history
            for side in ['left', 'right']:
                hip_name = f'{side}_hip'
                if hip_name in current_positions:
                    self.hip_history[side].append(current_positions[hip_name].copy())
                    if len(self.hip_history[side]) > self.hip_history_size:
                        self.hip_history[side].pop(0)

            self.position_history.append(current_positions)
            if len(self.position_history) > self.history_size:
                self.position_history.pop(0)

            # Smooth
            smoothed = self._smooth_all(water_level)

            for joint_name, pos in smoothed.items():
                frame_data[f'{joint_name}_x'] = round(pos['x'], 4)
                frame_data[f'{joint_name}_y'] = round(pos['y'], 4)
                frame_data[f'{joint_name}_z'] = round(pos['z'], 4)
                frame_data[f'{joint_name}_visibility'] = round(pos['visibility'], 4)
                frame_data[f'{joint_name}_above_water'] = pos['above_water']

        elif self.position_history and self.frames_since_detection <= SHORT_GAP_HOLD_FRAMES:
            # v4 — was `elif self.position_history:` with NO cap in v3,
            # holding the last position indefinitely on any sustained
            # loss and quietly making it look like tracking was still
            # working. Now only holds through brief gaps, with decaying
            # confidence; a real, sustained loss shows up as missing data
            # instead of being disguised.
            last = self.position_history[-1]
            decay = 0.6 ** self.frames_since_detection
            for joint_name, pos in last.items():
                frame_data[f'{joint_name}_x'] = round(pos['x'], 4)
                frame_data[f'{joint_name}_y'] = round(pos['y'], 4)
                frame_data[f'{joint_name}_z'] = round(pos.get('z', 0.0), 4)
                frame_data[f'{joint_name}_visibility'] = round(pos['visibility'] * decay, 4)
                frame_data[f'{joint_name}_above_water'] = pos['above_water']

        self.tracking_data.append(frame_data)
        return best_person

    def _filter_jumps(self, current_positions):
        """Reject individual landmarks that jump too far"""
        filtered = {}
        for joint_name, pos in current_positions.items():
            if joint_name in self.last_known:
                last = self.last_known[joint_name]
                dist = np.sqrt((pos['x'] - last['x'])**2 + (pos['y'] - last['y'])**2)
                if dist > self.max_jump:
                    filtered[joint_name] = {
                        'x': last['x'], 'y': last['y'],
                        'z': pos['z'],
                        'visibility': pos['visibility'] * 0.3,
                        'above_water': pos['above_water']
                    }
                    if 'source' in pos:
                        filtered[joint_name]['source'] = pos['source']
                    continue
            filtered[joint_name] = pos
            self.last_known[joint_name] = {'x': pos['x'], 'y': pos['y']}
        return filtered

    def _smooth_all(self, water_level):
        """Weighted temporal smoothing"""
        smoothed = {}

        def _weighted_avg(history, power=1.0):
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

        for joint_name in ALL_LANDMARKS:
            is_foot = 'foot' in joint_name or 'ankle' in joint_name or 'heel' in joint_name
            hist = [f[joint_name] for f in self.position_history if joint_name in f]
            if hist:
                s = _weighted_avg(hist, power=1.5 if is_foot else 1.0)
                smoothed[joint_name] = s

        # Hip smoothing — lighter than the general default so it follows
        # actual movement instead of lagging behind it
        for side in ['left', 'right']:
            if self.hip_history[side]:
                s = _weighted_avg(self.hip_history[side], power=1.5)
                smoothed[f'{side}_hip'] = s

        return smoothed

    def get_dataframe(self):
        return pd.DataFrame(self.tracking_data)


# ============================================================================
# VISUALIZATION
# ============================================================================

# COCO skeleton connections
COCO_CONNECTIONS = [
    (5, 6),    # shoulders
    (5, 7), (7, 9),     # left arm
    (6, 8), (8, 10),    # right arm
    (5, 11), (6, 12),   # torso sides
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

def draw_frame_rtmpose(frame, best_person, water_level, frame_num, total_frames):
    """Draw visualization with RTMPose keypoints"""
    viz = frame.copy()
    h, w = frame.shape[:2]
    water_y = int(water_level * h)

    # Waterline
    cv2.line(viz, (0, water_y), (w, water_y), (0, 0, 0), 4)
    cv2.line(viz, (0, water_y), (w, water_y), (0, 255, 255), 2)
    cv2.putText(viz, f"WATERLINE: {water_level:.3f}", (10, water_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.rectangle(viz, (10, 10), (550, 130), (0, 0, 0), -1)

    if best_person is not None:
        person_kps, person_scores = best_person

        # Draw skeleton connections
        for (i, j) in COCO_CONNECTIONS:
            if person_scores[i] > 0.2 and person_scores[j] > 0.2:
                pt1 = (int(person_kps[i][0]), int(person_kps[i][1]))
                pt2 = (int(person_kps[j][0]), int(person_kps[j][1]))
                thick = 4 if i >= 13 or j >= 13 else 3
                cv2.line(viz, pt1, pt2, (255, 255, 255), thick, cv2.LINE_AA)

        # Draw synthesized mid_spine point
        if all(person_scores[i] > 0.2 for i in [5, 6, 11, 12]):
            sc = ((person_kps[5][0] + person_kps[6][0]) / 2,
                  (person_kps[5][1] + person_kps[6][1]) / 2)
            hc = ((person_kps[11][0] + person_kps[12][0]) / 2,
                  (person_kps[11][1] + person_kps[12][1]) / 2)
            mid = (int((sc[0] + hc[0]) / 2), int((sc[1] + hc[1]) / 2))
            cv2.line(viz, (int(sc[0]), int(sc[1])), mid, (255, 200, 0), 2, cv2.LINE_AA)
            cv2.line(viz, mid, (int(hc[0]), int(hc[1])), (255, 200, 0), 2, cv2.LINE_AA)
            cv2.circle(viz, mid, 9, (255, 200, 0), -1)
            cv2.circle(viz, mid, 9, (255, 255, 255), 2)

        # Draw keypoints
        above = total_joints = 0
        for idx in range(17):
            if person_scores[idx] > 0.2:
                x, y = int(person_kps[idx][0]), int(person_kps[idx][1])
                is_above = y < water_y
                color = (0, 255, 0) if is_above else (0, 0, 255)
                radius = 12 if idx >= 15 else 8
                cv2.circle(viz, (x, y), radius, color, -1)
                cv2.circle(viz, (x, y), radius, (255, 255, 255), 2)
                above += int(is_above)
                total_joints += 1

        above_pct = (above / total_joints * 100) if total_joints > 0 else 0
        cv2.putText(viz, f"Above: {above}/{total_joints} ({above_pct:.0f}%)",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(viz, f"Below: {total_joints-above}/{total_joints} ({100-above_pct:.0f}%)",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(viz, "RTMPose-x UNDERWATER (locked+validated)",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    else:
        cv2.putText(viz, "No swimmer locked", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    ft = f"Frame: {frame_num}/{total_frames}"
    ts = cv2.getTextSize(ft, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(viz, ft, (550 - ts[0] - 10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return viz


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_video(video_path, output_path, water_level):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_fps = fps if OUTPUT_FPS is None else OUTPUT_FPS

    print(f"\n{'='*70}")
    print(f"RTMPose UNDERWATER TRACKER v4")
    print(f"{'='*70}")
    print(f"Video: {Path(video_path).name}")
    print(f"Resolution: {w}x{h} | FPS: {fps:.0f}")
    print(f"Frames: {total} | Duration: {total/fps:.1f}s")
    print(f"AI Model: RTMPose-x (384x288) + YOLOX-x detector | det_frequency={DET_FREQUENCY}")
    print(f"Locking: anatomy-validated, {RELOCK_CHECK_INTERVAL}f re-check, 2-in-a-row before switching")

    tracker = RTMPoseUnderwaterTracker()

    if water_level is None:
        water_level = detect_underwater_waterline(video_path)
        water_level = calibrate_waterline_to_shoulders(
            video_path, water_level, tracker.pose_tracker
        )
    else:
        print(f"\nWaterline: {water_level:.3f} (MANUAL)")

    max_frames_to_track = int(MAX_FIGURE_DURATION * fps) if MAX_FIGURE_DURATION else total
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
            print(f"\n  Stopping tracking at frame {frame_count}")
            tracking_stopped = True

        if not tracking_stopped:
            best_person = tracker.process_frame(frame, frame_count, fps, water_level)
        else:
            best_person = None

        annotated = draw_frame_rtmpose(frame, best_person, water_level,
                                        frame_count, total)
        out.write(annotated)
        frame_count += 1

        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_count / elapsed
            eta = (total - frame_count) / fps_proc if fps_proc > 0 else 0
            print(f"  {frame_count/total*100:.1f}% | {frame_count}/{total} | "
                  f"{fps_proc:.1f} fps | ETA: {eta:.0f}s")

    cap.release()
    out.release()
    time.sleep(0.5)

    elapsed_total = time.time() - start_time
    file_size = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*70}")
    print(f"COMPLETE! {total} frames in {elapsed_total:.1f}s")
    print(f"  Output: {output_path}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"{'='*70}\n")

    df = tracker.get_dataframe()
    csv_path = output_path.parent / f"{output_path.stem}_data.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Data saved: {csv_path}\n")

    return output_path, csv_path


# ============================================================================
# KALMAN FILTER (unchanged from v3 — deliberately looser than above-water's,
# tuned for underwater's faster/more erratic thrust movement)
# ============================================================================

class ImprovedKalmanFilter1D:
    def __init__(self, process_var=0.005, measurement_var=0.03, outlier_threshold=0.20):
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
        R_adj = self.R * (2.0 - confidence)
        y = measurement - (self.H @ self.x)[0]
        S = self.H @ self.P @ self.H.T + R_adj
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
            return np.nan if self.consecutive_predictions > self.max_predictions else predicted


def apply_kalman_filter_to_csv(csv_path):
    """Same numeric approach as v3 — for the faster numpy-array version
    (verified numerically identical, ~3x faster), see the web app's
    tracker_core.py, which already got that optimization."""
    print(f"\n{'='*70}")
    print(f"KALMAN FILTERING")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path)
    print(f"  Frames: {len(df)}")

    joints = ALL_LANDMARKS

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

            col_values = df[col].to_numpy()
            vis_values = df[vis_col].to_numpy() if vis_col in df.columns else None

            filtered = [None] * len(df)
            for i in range(len(df)):
                raw_m = col_values[i]
                m = None if pd.isna(raw_m) else raw_m
                if vis_values is not None:
                    raw_c = vis_values[i]
                    c = 1.0 if pd.isna(raw_c) else raw_c
                else:
                    c = 1.0
                if m is not None and kf.is_outlier(m):
                    m = None
                filtered[i] = kf.filter(m, c)

            df[f'{joint}_{axis}_raw'] = df[col]
            df[col] = filtered

    output_path = Path(csv_path).parent / (Path(csv_path).stem.replace('_data', '') + '_KALMAN.csv')
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"  Saved: {output_path.name}\n")
    return output_path


# ============================================================================
# BATCH RUN
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"RTMPose UNDERWATER TRACKER v4")
    print(f"{'='*70}")
    print(f"  AI: RTMPose-x (performance) + YOLOX-x detector")
    print(f"  Detection: every frame (det_frequency={DET_FREQUENCY})")
    print(f"  Locking: anatomy-validated, was fresh-every-frame in v3")
    print(f"{'='*70}\n")

    video_list = []
    if isinstance(VIDEO_PATHS, str):
        video_list = sorted(glob.glob(VIDEO_PATHS)) if '*' in VIDEO_PATHS or '?' in VIDEO_PATHS else [VIDEO_PATHS]
    elif isinstance(VIDEO_PATHS, list):
        video_list = VIDEO_PATHS

    existing = [v for v in video_list if Path(v).exists()]
    if not existing:
        print("No videos found")
        exit(1)

    print(f"Processing {len(existing)} video(s)...\n")
    results = []
    failed = []

    for i, vp in enumerate(existing, 1):
        video_path = Path(vp)
        output_path = video_path.parent / f"{video_path.stem}_UNDERWATER_tracking.mp4"

        print(f"\n{'#'*70}")
        print(f"VIDEO {i}/{len(existing)}: {video_path.name}")
        print(f"{'#'*70}")

        try:
            vf, cf = process_video(str(video_path), output_path, WATERLINE)
            if vf is None:
                failed.append({'video': video_path.name, 'error': 'Failed'})
                continue
            kcsv = apply_kalman_filter_to_csv(cf)
            results.append({'video': video_path.name, 'kalman_csv': kcsv})
            print(f"\nVideo {i}/{len(existing)} complete!")
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append({'video': video_path.name, 'error': str(e)})

    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(results)}/{len(existing)} succeeded")
    for r in results:
        print(f"  OK {r['video']} -> {r['kalman_csv'].name}")
    for f in failed:
        print(f"  FAIL {f['video']}: {f['error']}")
    print(f"{'='*70}\n")

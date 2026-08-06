#!/usr/bin/env python3
"""
BARRACUDA FIGURE SCORER — FINA Height-Based + Blended Deductions
=================================================================
Scoring approach:
  1. HEIGHT establishes a BASE SCORE from the FINA height chart
  2. DEDUCTIONS are 70% absolute (FINA standard) + 30% relative (group rank)

Deduction categories:
  1. Vertical alignment  — body tilt during ascent and descent (above water)
  2. Backpike            — body line post-peak
  3. Leg extension       — knee bend (computed and displayed, NOT currently
                            counted in the total — see _deduction_keys)

This is the original, simpler scoring logic (reverted from a fuller
above+underwater version that added ankle extension, underwater leg
extension, toe depth, layout timing, hip depth, head tuck, back roundness,
head crown, and hinging — those extra categories introduced scores that
didn't match expectations, so this version goes back to just the three
categories above).

USAGE:
    scorer = BarracudaScorer('/Users/mona/Desktop/Science fairs/Science fair 2026/Barracuda folders/Jmeet figures.nosync')
    scorer.score_all()
    scorer.print_summary_table()   # clean console table
    scorer.save_html_report()      # styled HTML, saved to scoring_results_with_html/
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


class BarracudaScorer:

    ABSOLUTE_WEIGHT = 0.70
    RELATIVE_WEIGHT = 0.30

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.figures = {}
        self.results = {}
        self._find_figures()

    # ── Figure discovery ──

    _LEGACY_ABOVE_RE = re.compile(
        r'^(?P<name>.+)_above_(?P<variant>POSE|FAST)_tracking(?:_data)?(?P<kalman>_KALMAN)?\.csv$')
    _LEGACY_BELOW_RE = re.compile(
        r'^(?P<name>.+)_below_(?P<variant>UNDERWATER)_tracking(?:_data)?(?P<kalman>_KALMAN)?\.csv$')

    _ABOVE_VARIANT_RANK = {('POSE', True): 0, ('POSE', False): 1,
                            ('FAST', True): 2, ('FAST', False): 3}
    _BELOW_VARIANT_RANK = {('UNDERWATER', True): 0, ('UNDERWATER', False): 1}

    def _find_figures_legacy_naming(self):
        above_candidates = {}
        below_candidates = {}

        for path in self.data_dir.rglob('*.csv'):
            m = self._LEGACY_ABOVE_RE.match(path.name)
            if m:
                name = m.group('name')
                variant = m.group('variant')
                kalman = bool(m.group('kalman'))
                rank = self._ABOVE_VARIANT_RANK[(variant, kalman)]
                above_candidates.setdefault(name, []).append((rank, path))
                continue
            m = self._LEGACY_BELOW_RE.match(path.name)
            if m:
                name = m.group('name')
                variant = m.group('variant')
                kalman = bool(m.group('kalman'))
                rank = self._BELOW_VARIANT_RANK[(variant, kalman)]
                below_candidates.setdefault(name, []).append((rank, path))

        if not above_candidates:
            return False

        print(f"  Using legacy naming convention (POSE/FAST/UNDERWATER "
              f"+ optional _KALMAN).")
        print(f"  Variant preference: POSE+KALMAN > POSE > FAST+KALMAN > FAST "
              f"(above), UNDERWATER+KALMAN > UNDERWATER (below).\n")

        for name in sorted(above_candidates.keys()):
            candidates = sorted(above_candidates[name], key=lambda c: c[0])
            best_rank, best_path = candidates[0]
            below_path = None
            if name in below_candidates:
                below_candidates[name].sort(key=lambda c: c[0])
                below_path = below_candidates[name][0][1]

            self.figures[name] = {'above': best_path, 'below': below_path}

            if len(candidates) > 1:
                skipped = ', '.join(p.name for _, p in candidates[1:])
                print(f"  Found: {name}")
                print(f"    above -> {best_path.name}  (also available, not used: {skipped})")
            else:
                print(f"  Found: {name}")
                print(f"    above -> {best_path.name}")
            print(f"    below -> {below_path.name if below_path else '(none found)'}")

        print(f"\n  Total: {len(self.figures)} figures (legacy naming)\n")
        return True

    def _find_figures(self):
        if not self.data_dir.exists():
            print(f"  ⚠ Directory does not exist: {self.data_dir}")
            print(f"    Double-check the path — it must point to the folder")
            print(f"    containing your *_above_tracking_data.csv files.\n")
            return

        above_files = sorted(self.data_dir.glob('*_above_tracking_data.csv'))
        searched_recursively = False
        if not above_files:
            above_files = sorted(self.data_dir.rglob('*_above_tracking_data.csv'))
            searched_recursively = True

        for ab_path in above_files:
            name = ab_path.name.replace('_above_tracking_data.csv', '')
            uw_path = ab_path.parent / f'{name}_below_tracking_data.csv'
            if name in self.figures:
                print(f"  ⚠ Duplicate figure name '{name}' found at {ab_path} "
                      f"— keeping the first one found, skipping this one.")
                continue
            self.figures[name] = {
                'above': ab_path,
                'below': uw_path if uw_path.exists() else None
            }
            print(f"  Found: {name}" + (f"  ({ab_path.parent})" if searched_recursively else ""))
        print(f"  Total: {len(self.figures)} figures"
              + (" (found via recursive subfolder search)\n" if searched_recursively and self.figures else "\n"))

        if not self.figures:
            if self._find_figures_legacy_naming():
                return

            print(f"  ⚠ No '*_above_tracking_data.csv' files found in:")
            print(f"    {self.data_dir}")
            print(f"    (searched this folder and all subfolders)")
            print(f"    Check that this is the same folder your tracker script")
            print(f"    writes its output CSVs to, and that filenames end in")
            print(f"    '_above_tracking_data.csv' / '_below_tracking_data.csv'.\n")

            try:
                entries = sorted(self.data_dir.iterdir())
            except Exception as e:
                entries = []
                print(f"    (couldn't list directory contents: {e})\n")

            if entries:
                print(f"    Contents of {self.data_dir}:")
                for e in entries[:25]:
                    tag = '/' if e.is_dir() else ''
                    print(f"      {e.name}{tag}")
                if len(entries) > 25:
                    print(f"      ... and {len(entries) - 25} more")
                print()

                csv_like = [e for e in entries if e.suffix.lower() == '.csv']
                if csv_like:
                    print(f"    Note: this folder DOES contain {len(csv_like)} CSV file(s),")
                    print(f"    but none end in '_above_tracking_data.csv'. If your tracker")
                    print(f"    uses different naming, either rename the files or update")
                    print(f"    the pattern in _find_figures().\n")
            else:
                print(f"    (folder exists but is empty)\n")

    # ── Helpers ──

    def _avg_lr(self, row, name):
        ly, ry = row.get(f'left_{name}_y'), row.get(f'right_{name}_y')
        lv = row.get(f'left_{name}_visibility')
        rv = row.get(f'right_{name}_visibility')
        l_ok = pd.notna(ly) and (pd.isna(lv) or lv > 0.1)
        r_ok = pd.notna(ry) and (pd.isna(rv) or rv > 0.1)
        if l_ok and r_ok: return (ly + ry) / 2
        elif l_ok: return ly
        elif r_ok: return ry
        return None

    def _avg_lr_x(self, row, name):
        lx, rx = row.get(f'left_{name}_x'), row.get(f'right_{name}_x')
        if pd.notna(lx) and pd.notna(rx): return (lx + rx) / 2
        elif pd.notna(lx): return lx
        elif pd.notna(rx): return rx
        return None

    def _collect(self, df, joint):
        return [v for v in (self._avg_lr(df.iloc[i], joint)
                for i in range(len(df))) if v is not None]

    def _joint_angle(self, p1, p2, p3):
        v1 = (p1[0]-p2[0], p1[1]-p2[1])
        v2 = (p3[0]-p2[0], p3[1]-p2[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        m1 = np.sqrt(v1[0]**2 + v1[1]**2)
        m2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if m1 < 0.001 or m2 < 0.001: return 180.0
        return np.degrees(np.arccos(np.clip(dot/(m1*m2), -1, 1)))

    # ── Height chart → base score ──

    def _height_base_score(self, foot_clearance):
        breakpoints = [
            (0.33, 10.0),
            (0.30, 9.5),
            (0.27, 9.0),
            (0.24, 8.5),
            (0.21, 8.0),
            (0.18, 7.5),
            (0.15, 7.0),
            (0.12, 6.5),
            (0.09, 6.0),
            (0.06, 5.0),
            (0.03, 4.0),
            (0.00, 3.0),
        ]
        if foot_clearance >= breakpoints[0][0]:
            return breakpoints[0][1]
        if foot_clearance <= breakpoints[-1][0]:
            return breakpoints[-1][1]
        for i in range(len(breakpoints) - 1):
            cl_hi, sc_hi = breakpoints[i]
            cl_lo, sc_lo = breakpoints[i + 1]
            if foot_clearance >= cl_lo:
                t = (foot_clearance - cl_lo) / (cl_hi - cl_lo)
                return sc_lo + t * (sc_hi - sc_lo)
        return 3.0

    # ── Absolute deduction scales ──

    def _abs_vertical_alignment(self, tilt):
        if tilt is None: return 0.5
        if tilt <= 3:    return 0.0
        elif tilt <= 5:  return 0.2
        elif tilt <= 7:  return 0.4
        elif tilt <= 9:  return 0.6
        elif tilt <= 12: return 0.8
        else:            return 1.0

    def _abs_backpike(self, bp):
        if bp >= 45:    return 1.0
        elif bp >= 30:  return 0.8
        elif bp >= 20:  return 0.5
        elif bp >= 10:  return 0.3
        elif bp >= 5:   return 0.2
        else:           return 0.0

    def _abs_leg_extension(self, knee):
        if knee is None: return 0.3
        if knee >= 175:   return 0.0
        elif knee >= 168: return 0.2
        elif knee >= 160: return 0.4
        elif knee >= 150: return 0.6
        else:             return 1.0

    # ── Relative deduction ──

    def _relative_deduction(self, value, all_values, max_deduction, higher_is_worse=True):
        valid = [v for v in all_values if v is not None]
        if not valid or value is None:
            return 0.0

        if len(valid) == 1:
            return 0.0

        best = min(valid) if higher_is_worse else max(valid)
        worst = max(valid) if higher_is_worse else min(valid)

        if best == worst:
            return 0.0

        if higher_is_worse:
            t = (value - best) / (worst - best)
        else:
            t = (best - value) / (best - worst)

        t = np.clip(t, 0, 1)
        return round(t * max_deduction, 2)

    # ── Measurements ──

    def _extract_measurements(self, name):
        paths = self.figures[name]
        ab = pd.read_csv(paths['above'])
        uw = pd.read_csv(paths['below']) if paths['below'] else None
        wl_ab = ab['water_level'].median()

        m = {'name': name, 'frames': len(ab)}

        if len(ab) > 1:
            dt = ab.iloc[1]['time_seconds'] - ab.iloc[0]['time_seconds']
            m['fps'] = 1.0 / dt if dt > 0 else 30.0
        else:
            m['fps'] = 30.0

        ab_ankles = self._collect(ab, 'ankle')
        m['foot_clearance'] = (wl_ab - min(ab_ankles)) if ab_ankles else 0

        peak_frame = None
        if ab_ankles:
            min_ankle = min(ab_ankles)
            for i in range(len(ab)):
                a = self._avg_lr(ab.iloc[i], 'ankle')
                if a is not None and a == min_ankle:
                    peak_frame = i; break

        ascent_tilts = []
        knee_angles = []
        if peak_frame is not None:
            for fn in range(max(0, peak_frame - 7), min(len(ab), peak_frame + 8)):
                hx = self._avg_lr_x(ab.iloc[fn], 'hip')
                hy = self._avg_lr(ab.iloc[fn], 'hip')
                ax = self._avg_lr_x(ab.iloc[fn], 'ankle')
                ay = self._avg_lr(ab.iloc[fn], 'ankle')
                kx = self._avg_lr_x(ab.iloc[fn], 'knee')
                ky = self._avg_lr(ab.iloc[fn], 'knee')

                if all(v is not None for v in [hx, hy, ax, ay]):
                    dx = ax - hx; dy = ay - hy
                    ascent_tilts.append(abs(np.degrees(np.arctan2(dx, abs(dy)))))

                if all(v is not None for v in [hx, hy, kx, ky, ax, ay]):
                    knee_angles.append(self._joint_angle((hx, hy), (kx, ky), (ax, ay)))

        m['ascent_tilt_median'] = np.median(ascent_tilts) if ascent_tilts else None

        descent_tilts = []
        if peak_frame is not None:
            for fn in range(peak_frame, min(len(ab), peak_frame + 40)):
                hx = self._avg_lr_x(ab.iloc[fn], 'hip')
                hy = self._avg_lr(ab.iloc[fn], 'hip')
                ax = self._avg_lr_x(ab.iloc[fn], 'ankle')
                ay = self._avg_lr(ab.iloc[fn], 'ankle')

                if all(v is not None for v in [hx, hy, ax, ay]):
                    if ay < wl_ab:
                        dx = ax - hx; dy = ay - hy
                        descent_tilts.append(abs(np.degrees(np.arctan2(dx, abs(dy)))))

        m['descent_tilt_median'] = np.median(descent_tilts) if descent_tilts else None

        ascent = m['ascent_tilt_median']
        descent = m['descent_tilt_median']
        if ascent is not None and descent is not None:
            m['worst_tilt'] = max(ascent, descent)
        elif ascent is not None:
            m['worst_tilt'] = ascent
        elif descent is not None:
            m['worst_tilt'] = descent
        else:
            m['worst_tilt'] = None

        m['knee_angle_median'] = np.median(knee_angles) if knee_angles else None

        if peak_frame is not None:
            bp_angles = []
            for fn in range(peak_frame, min(len(ab), peak_frame + 40)):
                hx = self._avg_lr_x(ab.iloc[fn], 'hip')
                hy = self._avg_lr(ab.iloc[fn], 'hip')
                kx = self._avg_lr_x(ab.iloc[fn], 'knee')
                ky = self._avg_lr(ab.iloc[fn], 'knee')

                if all(v is not None for v in [hx, hy, kx, ky]):
                    if ky < wl_ab - 0.05:
                        dx = kx - hx
                        dy = hy - ky
                        if dy > 0.01:
                            bp_angles.append(abs(np.degrees(np.arctan2(dx, dy))))

            if bp_angles:
                m['backpike_worst'] = max(bp_angles)
                m['backpike_sustained'] = np.median(sorted(bp_angles, reverse=True)[:5])
                m['backpike_score'] = m['backpike_worst'] * 0.6 + m['backpike_sustained'] * 0.4
            else:
                m['backpike_worst'] = 0
                m['backpike_sustained'] = 0
                m['backpike_score'] = 0
        else:
            m['backpike_worst'] = 0
            m['backpike_sustained'] = 0
            m['backpike_score'] = 0

        if uw is not None:
            uw_hips = [(i, self._avg_lr(uw.iloc[i], 'hip'))
                       for i in range(len(uw))]
            uw_hips = [(i, h) for i, h in uw_hips if h is not None]

            if uw_hips:
                peak_i, peak_h = min(uw_hips, key=lambda x: x[1])

                threshold = peak_h + 0.10
                hold_frames = sum(1 for i, h in uw_hips
                                  if h <= threshold and i >= peak_i)
                m['hold_duration_sec'] = round(hold_frames / m['fps'], 2)

                post_peak = [(i, h) for i, h in uw_hips
                             if i > peak_i + 3 and i < peak_i + 25]
                if len(post_peak) >= 3:
                    m['descent_rate'] = (post_peak[-1][1] - post_peak[0][1]) / \
                                        (post_peak[-1][0] - post_peak[0][0])
                else:
                    m['descent_rate'] = 0.01

                hold_start = peak_i + 3
                hold_end = min(peak_i + 25, len(uw))
                hold_hips = [h for i, h in uw_hips if hold_start <= i < hold_end]
                if hold_hips:
                    m['hold_stability_std'] = np.std(hold_hips)
                else:
                    m['hold_stability_std'] = 0.15

        return m

    # ── Two-pass scoring: extract all, then compute blended deductions ──

    def score_all(self):
        if not self.figures:
            print("  ⚠ No figures loaded — nothing to score.")
            print("    Check the data_dir path passed to BarracudaScorer() and")
            print("    that it contains '*_above_tracking_data.csv' files.\n")
            return self.results

        measurements = {}
        for name in sorted(self.figures.keys()):
            measurements[name] = self._extract_measurements(name)

        all_ascent = [measurements[n].get('ascent_tilt_median') for n in measurements]
        all_descent = [measurements[n].get('descent_tilt_median') for n in measurements]
        all_bp = [measurements[n].get('backpike_score', 0) for n in measurements]
        all_knee = [measurements[n].get('knee_angle_median') for n in measurements]

        max_ded = 1.0

        for name in sorted(self.figures.keys()):
            m = measurements[name]
            d = {}

            ascent = m.get('ascent_tilt_median')
            abs_ded = self._abs_vertical_alignment(ascent)
            rel_ded = self._relative_deduction(
                ascent, all_ascent, max_ded, higher_is_worse=True)
            d['ascent_alignment'] = min(1.0, round(
                self.ABSOLUTE_WEIGHT * abs_ded + self.RELATIVE_WEIGHT * rel_ded, 2))
            d['ascent_alignment_abs'] = abs_ded
            d['ascent_alignment_rel'] = round(rel_ded, 2)
            d['ascent_alignment_degrees'] = round(ascent, 1) if ascent is not None else None

            descent = m.get('descent_tilt_median')
            abs_ded = self._abs_vertical_alignment(descent)
            rel_ded = self._relative_deduction(
                descent, all_descent, max_ded, higher_is_worse=True)
            d['descent_alignment'] = min(1.0, round(
                self.ABSOLUTE_WEIGHT * abs_ded + self.RELATIVE_WEIGHT * rel_ded, 2))
            d['descent_alignment_abs'] = abs_ded
            d['descent_alignment_rel'] = round(rel_ded, 2)
            d['descent_alignment_degrees'] = round(descent, 1) if descent is not None else None

            bp = m.get('backpike_score', 0)
            abs_ded = self._abs_backpike(bp)
            rel_ded = self._relative_deduction(
                bp, all_bp, max_ded, higher_is_worse=True)
            d['backpike'] = min(1.0, round(
                self.ABSOLUTE_WEIGHT * abs_ded + self.RELATIVE_WEIGHT * rel_ded, 2))
            d['backpike_abs'] = abs_ded
            d['backpike_rel'] = round(rel_ded, 2)
            d['backpike_degrees'] = round(bp, 1)

            knee = m.get('knee_angle_median')
            abs_ded = self._abs_leg_extension(knee)
            rel_ded = self._relative_deduction(
                knee, all_knee, max_ded, higher_is_worse=False)
            d['leg_extension'] = min(1.0, round(
                self.ABSOLUTE_WEIGHT * abs_ded + self.RELATIVE_WEIGHT * rel_ded, 2))
            d['leg_extension_abs'] = abs_ded
            d['leg_extension_rel'] = round(rel_ded, 2)
            d['leg_extension_degrees'] = round(knee, 1) if knee is not None else None

            base = self._height_base_score(m.get('foot_clearance', 0))
            m['base_score'] = round(base, 2)

            total_ded = sum(d.get(k, 0) for k in self._deduction_keys())
            m['deductions'] = d
            m['total_deduction'] = round(total_ded, 2)
            m['score'] = round(max(0.0, base - total_ded) * 20) / 20

            self.results[name] = m

        return self.results

    def _deduction_keys(self):
        return ['ascent_alignment', 'descent_alignment', 'backpike']

    def score_figure(self, name):
        """Score a single figure (without relative component).
        For full scoring with relative ranking, use score_all()."""
        m = self._extract_measurements(name)
        d = {}

        ascent = m.get('ascent_tilt_median')
        d['ascent_alignment'] = self._abs_vertical_alignment(ascent)
        d['ascent_alignment_degrees'] = round(ascent, 1) if ascent is not None else None

        descent = m.get('descent_tilt_median')
        d['descent_alignment'] = self._abs_vertical_alignment(descent)
        d['descent_alignment_degrees'] = round(descent, 1) if descent is not None else None

        bp = m.get('backpike_score', 0)
        d['backpike'] = self._abs_backpike(bp)
        d['backpike_degrees'] = round(bp, 1)

        knee = m.get('knee_angle_median')
        d['leg_extension'] = self._abs_leg_extension(knee)
        d['leg_extension_degrees'] = round(knee, 1) if knee is not None else None

        base = self._height_base_score(m.get('foot_clearance', 0))
        m['base_score'] = round(base, 2)

        total_ded = sum(d.get(k, 0) for k in self._deduction_keys())
        m['deductions'] = d
        m['total_deduction'] = round(total_ded, 2)
        m['score'] = round(max(0.0, base - total_ded) * 20) / 20

        self.results[name] = m
        return m

    @classmethod
    def score_single_pair(cls, above_csv_path, below_csv_path=None, name="figure"):
        """
        Score one figure directly from its above/below CSV paths, with no
        folder scanning. This is what the web app calls right after
        tracking finishes — it bypasses _find_figures() entirely so the
        exact CSV paths returned by tracker_core are used, regardless of
        their filenames.
        """
        scorer = cls.__new__(cls)
        scorer.data_dir = None
        scorer.figures = {
            name: {
                'above': Path(above_csv_path),
                'below': Path(below_csv_path) if below_csv_path else None,
            }
        }
        scorer.results = {}
        return scorer.score_figure(name)

    # ── Output ──

    def summary_dataframe(self):
        if not self.results:
            self.score_all()
        if not self.results:
            return pd.DataFrame()

        rows = []
        for name, r in self.results.items():
            d = r['deductions']

            labels = {'ascent_alignment': 'ascent align.',
                      'descent_alignment': 'descent align.',
                      'backpike': 'backpike'}
            top = sorted(
                [(k, d[k]) for k in self._deduction_keys() if d.get(k, 0) > 0],
                key=lambda x: x[1], reverse=True)
            top_str = ', '.join(f"{labels.get(k, k)} -{v:.2f}" for k, v in top) or '—'

            s = r['score']
            if s >= 9.5:   assess = 'Excellent'
            elif s >= 8.5: assess = 'Very Good'
            elif s >= 7.5: assess = 'Good'
            elif s >= 6.5: assess = 'Competent'
            elif s >= 5.5: assess = 'Satisfactory'
            elif s >= 4.5: assess = 'Deficient'
            else:          assess = 'Weak'

            rows.append({
                'Figure': name,
                'Score': s,
                'Assessment': assess,
                'Base': r['base_score'],
                'Deduction': r['total_deduction'],
                'Ascent°': r.get('ascent_tilt_median'),
                'Descent°': r.get('descent_tilt_median'),
                'Backpike°': r.get('backpike_score'),
                'Knee°': r.get('knee_angle_median'),
                'Foot Clr.': r.get('foot_clearance'),
                'Top Deductions': top_str,
            })

        df = pd.DataFrame(rows).sort_values('Score', ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = 'Rank'

        round_cols = ['Score', 'Base', 'Deduction', 'Ascent°', 'Descent°',
                      'Backpike°', 'Knee°', 'Foot Clr.']
        for c in round_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').round(2)
        return df

    def print_summary_table(self):
        df = self.summary_dataframe()
        if df.empty:
            print("  ⚠ Nothing to display — no figures were found or scored.")
            print(f"    data_dir: {self.data_dir}")
            return

        print(f"\n{'='*100}")
        print(f"  BARRACUDA SCORER — Summary ({len(df)} Figures)")
        print(f"{'='*100}\n")

        numeric_cols = ['Figure', 'Score', 'Assessment', 'Base', 'Deduction',
                         'Ascent°', 'Descent°', 'Backpike°', 'Knee°', 'Foot Clr.']
        with pd.option_context('display.max_colwidth', None,
                                'display.width', None):
            print(df[numeric_cols].to_string(na_rep='—'))

        print(f"\n{'-'*100}")
        print(f"  Top deductions per figure:")
        print(f"{'-'*100}")
        name_w = max(len(n) for n in df['Figure']) + 2
        for rank, row in df.iterrows():
            print(f"  {rank}. {row['Figure']:<{name_w}} {row['Top Deductions']}")

        print(f"\n{'='*100}")
        print(f"  Note: leg extension is measured and shown (see print_comparison())")
        print(f"  but not currently counted toward the score. Scores also exclude")
        print(f"  back pike / thrust quality (not measurable from tracker data).")
        print(f"{'='*100}\n")

    def save_html_report(self, output_path=None):
        df = self.summary_dataframe()
        if df.empty:
            print("  ⚠ Nothing to export — no figures were found or scored.")
            return None

        if output_path is None:
            out_dir = self.data_dir / 'scoring_results_with_html'
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / 'barracuda_summary.html'
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        styled = (
            df.style
            .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=10)
            .format({
                'Score': '{:.2f}', 'Base': '{:.2f}', 'Deduction': '{:.2f}',
                'Ascent°': '{:.1f}', 'Descent°': '{:.1f}',
                'Backpike°': '{:.1f}', 'Knee°': '{:.1f}',
                'Foot Clr.': '{:+.3f}',
            }, na_rep='—')
            .set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#2b2b2b'), ('color', 'white'),
                    ('padding', '8px 12px'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [
                    ('padding', '6px 12px'), ('text-align', 'center'),
                    ('font-family', 'Helvetica, Arial, sans-serif')]},
                {'selector': 'table', 'props': [
                    ('border-collapse', 'collapse'), ('margin', '20px auto')]},
            ])
            .set_properties(subset=['Figure', 'Top Deductions'], **{'text-align': 'left'})
        )

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Barracuda Scoring Summary</title></head>
<body style="font-family: Helvetica, Arial, sans-serif; background: #f5f5f5;">
<h2 style="text-align:center;">Barracuda Figure Scoring — {len(df)} Figures</h2>
{styled.to_html()}
<p style="text-align:center; color:#666; font-size:0.9em;">
Leg extension is measured but not counted toward the score. Scores also
exclude back pike / thrust quality (not measurable from tracker data).
</p>
</body></html>"""

        output_path.write_text(html)
        print(f"  ✓ HTML report saved: {output_path}")
        return output_path


if __name__ == "__main__":
    data_dir = '/Users/mona/Desktop/Science fairs/Science fair 2026/Barracuda folders/Jmeet figures.nosync'
    print(f"\n{'='*80}")
    print(f"  BARRACUDA FIGURE SCORER — FINA-Aligned Deductions")
    print(f"  Directory: {data_dir}")
    print(f"{'='*80}\n")
    scorer = BarracudaScorer(data_dir)
    scorer.score_all()
    scorer.print_summary_table()
    scorer.save_html_report()

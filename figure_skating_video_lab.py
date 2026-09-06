import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Figure Skating Video Lab", page_icon="⛸️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: .8rem;}
.hero {background:linear-gradient(135deg,#0f172a,#1d4ed8 55%,#38bdf8);color:white;padding:1.25rem;border-radius:24px;}
.card {background:#fff;border:1px solid #e2e8f0;border-radius:16px;padding:1rem;margin:.3rem 0;}
.warning {background:#fff7ed;border:1px solid #fdba74;border-radius:12px;padding:.8rem;}
</style>
""", unsafe_allow_html=True)

LANDMARKS = {
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
]


def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1))))


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "frames": count, "width": width, "height": height, "duration": count / fps if fps else 0}


def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_summary(path, sample_count=20):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    positions = np.linspace(0, max(total - 1, 0), min(sample_count, max(total, 1))).astype(int)
    scores = []
    prev = None
    motion = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            continue
        scores.append(blur_score(frame))
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if prev is not None:
            motion.append(float(np.mean(cv2.absdiff(small, prev))))
        prev = small
    cap.release()
    return {
        "blur": round(float(np.mean(scores)) if scores else 0, 1),
        "camera_motion": round(float(np.mean(motion)) if motion else 0, 2),
    }


def get_pose_backend():
    try:
        import mediapipe as mp
        try:
            from mediapipe.python.solutions import pose as mp_pose
        except Exception:
            mp_pose = mp.solutions.pose
        return mp_pose, None
    except Exception as exc:
        return None, str(exc)


def draw_skeleton(frame, landmarks, visibility_threshold=0.55):
    h, w = frame.shape[:2]
    points = {}
    for idx, lm in enumerate(landmarks):
        if lm.visibility >= visibility_threshold:
            p = (int(lm.x * w), int(lm.y * h))
            points[idx] = p
            cv2.circle(frame, p, 4, (0, 220, 255), -1)
    for a, b in CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (255, 230, 0), 2)
    return frame, points


def analyze_video(video_path, max_frames, frame_stride, threshold):
    mp_pose, error = get_pose_backend()
    if mp_pose is None:
        return None, error

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    processed = 0
    current = 0
    rows = []
    annotated = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=threshold,
        min_tracking_confidence=threshold,
    ) as pose:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if current % frame_stride != 0:
                current += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            row = {"frame": current, "time_s": round(current / fps, 3), "pose_confidence": 0.0, "knee_angle": np.nan, "trunk_angle": np.nan, "center_x": np.nan, "center_y": np.nan}
            out = frame.copy()
            if result.pose_landmarks:
                lms = result.pose_landmarks.landmark
                visible = [lm.visibility for lm in lms]
                row["pose_confidence"] = float(np.mean(visible))
                out, pts = draw_skeleton(out, lms, threshold)
                needed_left = [LANDMARKS["LEFT_HIP"], LANDMARKS["LEFT_KNEE"], LANDMARKS["LEFT_ANKLE"]]
                needed_right = [LANDMARKS["RIGHT_HIP"], LANDMARKS["RIGHT_KNEE"], LANDMARKS["RIGHT_ANKLE"]]
                candidates = []
                for triplet in [needed_left, needed_right]:
                    if all(i in pts for i in triplet):
                        candidates.append(angle(pts[triplet[0]], pts[triplet[1]], pts[triplet[2]]))
                if candidates:
                    row["knee_angle"] = float(np.nanmean(candidates))
                shoulders = [LANDMARKS["LEFT_SHOULDER"], LANDMARKS["RIGHT_SHOULDER"]]
                hips = [LANDMARKS["LEFT_HIP"], LANDMARKS["RIGHT_HIP"]]
                if all(i in pts for i in shoulders + hips):
                    shoulder_mid = np.mean([pts[i] for i in shoulders], axis=0)
                    hip_mid = np.mean([pts[i] for i in hips], axis=0)
                    vector = shoulder_mid - hip_mid
                    row["trunk_angle"] = float(abs(np.degrees(np.arctan2(vector[0], -vector[1]))))
                    row["center_x"], row["center_y"] = float(hip_mid[0] / width), float(hip_mid[1] / height)
                cv2.putText(out, f"confidence: {row['pose_confidence']:.2f}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(out, "No reliable pose", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            rows.append(row)
            annotated.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            processed += 1
            current += 1
    cap.release()
    df = pd.DataFrame(rows)
    return {"data": df, "frames": annotated, "fps": fps, "width": width, "height": height}, None


def infer_movement(df):
    usable = df.dropna(subset=["pose_confidence"])
    coverage = float((usable["pose_confidence"] >= 0.55).mean()) if len(usable) else 0.0
    if coverage < 0.25:
        return [("Unknown / needs coach review", 0.15), ("Jump attempt", 0.10), ("Spin", 0.10)]
    cy = usable["center_y"].dropna()
    cx = usable["center_x"].dropna()
    knee = usable["knee_angle"].dropna()
    vertical_range = float(cy.max() - cy.min()) if len(cy) else 0.0
    horizontal_range = float(cx.max() - cx.min()) if len(cx) else 0.0
    knee_var = float(knee.std()) if len(knee) > 2 else 0.0
    jump = min(0.80, 0.25 + vertical_range * 2.2 + knee_var / 120)
    spin = min(0.75, 0.20 + max(0, 0.25 - horizontal_range) + max(0, 18 - knee_var) / 100)
    steps = min(0.70, 0.15 + horizontal_range * 0.9 + knee_var / 150)
    values = [("Jump attempt", jump), ("Spin", spin), ("Step sequence", steps)]
    total = sum(v for _, v in values)
    return sorted([(name, round(v / total, 2)) for name, v in values], key=lambda x: x[1], reverse=True)


def build_observations(df, metadata, quality):
    observations = []
    coverage = float((df["pose_confidence"] >= 0.55).mean()) if len(df) else 0.0
    if coverage < 0.55:
        observations.append({"observation": "Low body-visibility coverage / ظهور الجسم غير كافٍ", "confidence": "Low", "evidence": f"Reliable pose detected in {coverage:.0%} of analyzed frames.", "limit": "Some joints are missing or occluded; do not infer detailed technique from hidden frames.", "fix": "Record again from a stable side angle with the full body visible."})
    knee = df["knee_angle"].dropna()
    trunk = df["trunk_angle"].dropna()
    if len(knee) > 5 and knee.std() > 28:
        observations.append({"observation": "Variable lower-body position / تباين واضح في وضع الركبة", "confidence": "Medium", "evidence": f"Knee-angle variability: {knee.std():.1f}° across tracked frames.", "limit": "2D camera angle changes apparent joint angles.", "fix": "On-ice: controlled entry-to-landing checks. Off-ice: single-leg control. Drill: 3×5 stick-landings with a two-second hold."})
    if len(trunk) > 5 and trunk.mean() > 28:
        observations.append({"observation": "Possible trunk lean / احتمال ميل زائد للجذع", "confidence": "Medium", "evidence": f"Average visual trunk lean: {trunk.mean():.1f}°.", "limit": "This is a 2D visual estimate, not a clinical or biomechanical diagnosis.", "fix": "On-ice: quiet-shoulder entries. Off-ice: anti-rotation core work. Drill: mirror alignment holds before takeoff."})
    if quality["blur"] < 50:
        observations.append({"observation": "Video blur limits precision / ضبابية الفيديو تحد من الدقة", "confidence": "High", "evidence": f"Blur metric: {quality['blur']:.1f}.", "limit": "Fast foot and rotation timing cannot be assessed reliably.", "fix": "Use higher shutter speed, brighter lighting, and 60 fps or above when available."})
    if metadata["fps"] and metadata["fps"] < 50:
        observations.append({"observation": "Low frame-rate for flight/rotation timing / معدل إطارات منخفض", "confidence": "High", "evidence": f"Video FPS: {metadata['fps']:.1f}.", "limit": "Do not make a precise rotation-count or under-rotation claim from this clip alone.", "fix": "Record future jump clips at 60 fps or higher from a stable side angle."})
    if not observations:
        observations.append({"observation": "No high-confidence technical flag found / لا توجد ملاحظة فنية عالية الثقة", "confidence": "Low", "evidence": "Tracked motion appears internally consistent in the available 2D footage.", "limit": "Absence of a flag is not proof of correct technique.", "fix": "Coach review recommended; compare with a baseline clip from the same camera angle."})
    return observations


def confidence_label(coverage, metadata, quality):
    score = 0.55 * coverage + 0.25 * min(metadata["fps"] / 60, 1) + 0.20 * min(quality["blur"] / 120, 1)
    if score >= 0.75:
        return "High", round(score, 2)
    if score >= 0.50:
        return "Medium", round(score, 2)
    return "Low", round(score, 2)


st.markdown('<div class="hero"><h1>⛸️ Figure Skating Video Lab</h1><p>رفع فيديو، هيكل عظمي متحرك، تحليل احتمالي للحركة، وأخطاء فنية مدعومة بالبيانات وحدود واضحة للّقطة.</p></div>', unsafe_allow_html=True)
st.warning("هذا المختبر أداة مساعدة للمدرب. النتائج مبنية على فيديو ثنائي الأبعاد ولا تمثل حكم ISU رسميًا أو تشخيصًا طبيًا.")

with st.sidebar:
    st.header("Analysis Setup / إعداد التحليل")
    athlete = st.text_input("Athlete / اللاعب", "Skater")
    expected = st.selectbox("Expected element (optional) / العنصر المتوقع", ["Unknown", "Jump", "Spin", "Step sequence"])
    camera_angle = st.selectbox("Camera angle / زاوية الكاميرا", ["Side", "Front", "Diagonal", "Unknown"])
    max_frames = st.slider("Maximum processed frames / الحد الأقصى للإطارات", 60, 600, 240, 30)
    stride = st.slider("Analyze every Nth frame / تحليل كل N إطار", 1, 6, 2)
    threshold = st.slider("Landmark confidence threshold / عتبة الثقة", 0.30, 0.85, 0.55, 0.05)

video = st.file_uploader("Upload skating video / ارفع فيديو التزلج", type=["mp4", "mov", "avi"])

if video is None:
    st.info("Upload a video to begin. Best result: full body, stable side angle, bright lighting, and 60 fps or more for jumps.")
    st.stop()

suffix = Path(video.name).suffix or ".mp4"
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
tmp.write(video.getbuffer())
tmp.close()
video_path = Path(tmp.name)
metadata = video_metadata(video_path)
quality = quality_summary(video_path)

st.subheader("1. Clip quality / جودة اللقطة")
a, b, c, d = st.columns(4)
a.metric("Resolution", f"{metadata['width']}×{metadata['height']}")
b.metric("FPS", f"{metadata['fps']:.1f}")
c.metric("Duration", f"{metadata['duration']:.1f} s")
d.metric("Frames", metadata["frames"])

quality_notes = []
if metadata["fps"] < 50:
    quality_notes.append("Low FPS: precise timing, rotation count, and under-rotation cannot be concluded.")
if camera_angle == "Front":
    quality_notes.append("Front angle: edge depth and takeoff path are difficult to assess.")
if camera_angle == "Unknown":
    quality_notes.append("Camera angle not specified: all technique interpretations should be reviewed carefully.")
if quality["blur"] < 50:
    quality_notes.append("High blur detected: fast foot and air-position details may be unreliable.")
if quality["camera_motion"] > 18:
    quality_notes.append("Camera movement is noticeable: center-path measurements may be less reliable.")
for note in quality_notes or ["Clip quality has no automatic critical flag; full-body visibility still needs verification during pose tracking."]:
    st.info(note)

st.subheader("2. Original video / الفيديو الأصلي")
st.video(video.getvalue())

if st.button("🔬 Run video analysis / ابدأ تحليل الفيديو", type="primary", use_container_width=True):
    with st.spinner("Extracting pose landmarks and movement features… / يجري استخراج نقاط الحركة والمؤشرات…"):
        result, backend_error = analyze_video(video_path, max_frames, stride, threshold)
    if result is None:
        st.error("Pose-analysis backend is unavailable. The app remains usable for manual video review, but skeleton analysis needs a compatible MediaPipe/Python installation.")
        st.code(f"Backend detail: {backend_error}")
        st.stop()

    df = result["data"]
    coverage = float((df["pose_confidence"] >= threshold).mean()) if len(df) else 0.0
    label, confidence = confidence_label(coverage, metadata, quality)
    classes = infer_movement(df)
    observations = build_observations(df, metadata, quality)

    st.session_state["lab"] = {
        "df": df,
        "frames": result["frames"],
        "coverage": coverage,
        "confidence": confidence,
        "confidence_label": label,
        "classes": classes,
        "observations": observations,
        "metadata": metadata,
        "quality": quality,
        "athlete": athlete,
        "expected": expected,
        "camera_angle": camera_angle,
    }

lab = st.session_state.get("lab")
if not lab:
    st.stop()

st.subheader("3. Data confidence and clip limits / الثقة وحدود اللقطة")
a, b, c = st.columns(3)
a.metric("Overall confidence / الثقة العامة", f"{lab['confidence_label']} ({lab['confidence']:.2f})")
b.metric("Reliable-pose coverage / تغطية الهيكل الموثوق", f"{lab['coverage']:.0%}")
c.metric("Camera angle / زاوية الكاميرا", lab["camera_angle"])
st.caption("Confidence combines pose coverage, frame rate, and blur. It is not a probability that the technical conclusion is correct.")

st.subheader("4. Annotated skeleton review / مراجعة الهيكل العظمي")
frames = lab["frames"]
if frames:
    position = st.slider("Reviewed processed frame / إطار المراجعة", 0, len(frames) - 1, 0)
    left, right = st.columns(2)
    with left:
        st.image(frames[position], caption=f"Skeleton overlay — processed frame {position}", use_container_width=True)
    with right:
        row = lab["df"].iloc[position]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"**Time:** {row['time_s']:.2f}s")
        st.write(f"**Pose confidence:** {row['pose_confidence']:.2f}")
        st.write(f"**Visual knee angle:** {row['knee_angle']:.1f}°" if pd.notna(row['knee_angle']) else "**Visual knee angle:** unavailable")
        st.write(f"**Visual trunk lean:** {row['trunk_angle']:.1f}°" if pd.notna(row['trunk_angle']) else "**Visual trunk lean:** unavailable")
        st.caption("Angles are 2D image-based estimates and must be interpreted by the coach.")
        st.markdown('</div>', unsafe_allow_html=True)

st.subheader("5. Movement features / مؤشرات الحركة")
df = lab["df"]
chart = df.set_index("time_s")[["pose_confidence", "knee_angle", "trunk_angle"]]
st.line_chart(chart, height=280)

st.subheader("6. Probable movement type / نوع الحركة المرجّح")
for i, (name, score) in enumerate(lab["classes"]):
    if i == 0:
        st.success(f"Most likely: {name} — confidence proxy {score:.2f}")
    else:
        st.write(f"Alternative: {name} — {score:.2f}")
st.caption("This is a broad movement-category suggestion only. It does not identify a specific jump, rotation count, edge, level, or official technical call.")

st.subheader("7. Technical observations and practical corrections / الملاحظات الفنية والإصلاح")
for obs in lab["observations"]:
    st.markdown(f"<div class='card'><b>{obs['observation']}</b><br><br><b>Confidence:</b> {obs['confidence']}<br><b>Evidence:</b> {obs['evidence']}<br><b>Limit:</b> {obs['limit']}<br><b>Practical next step:</b> {obs['fix']}</div>", unsafe_allow_html=True)

report = {
    "athlete": lab["athlete"],
    "expected_element": lab["expected"],
    "camera_angle": lab["camera_angle"],
    "video_metadata": lab["metadata"],
    "clip_quality": lab["quality"],
    "data_confidence": {"label": lab["confidence_label"], "score": lab["confidence"], "reliable_pose_coverage": lab["coverage"]},
    "probable_movement_categories": lab["classes"],
    "observations": lab["observations"],
    "limitations": [
        "2D video-derived visual estimates only.",
        "Not an official ISU technical call or score.",
        "Not a medical diagnosis.",
        "Coach review is required before training or competition decisions.",
    ],
}
st.download_button("Download lab report JSON / تحميل تقرير المختبر", json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"), "figure_skating_video_lab_report.json", "application/json")
st.download_button("Download movement data CSV / تحميل بيانات الحركة", df.to_csv(index=False).encode("utf-8"), "figure_skating_motion_data.csv", "text/csv")

try:
    os.unlink(video_path)
except OSError:
    pass

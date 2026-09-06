import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import io
import json
from datetime import datetime

st.set_page_config(
    page_title="Pro Analyzer Advanced",
    page_icon="⛸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
.metric-card {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #dbeafe;
    padding: 1rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
}
.score-good {color:#15803d; font-weight:700;}
.score-mid {color:#b45309; font-weight:700;}
.score-bad {color:#b91c1c; font-weight:700;}
.small-note {color:#475569; font-size:0.95rem;}
</style>
""", unsafe_allow_html=True)

MOVEMENT_STANDARDS = {
    "single_axel": {"type": "jump", "base_value": 1.10, "ideal_knee_min": 125, "ideal_knee_max": 165, "ideal_trunk": 90},
    "double_axel": {"type": "jump", "base_value": 3.30, "ideal_knee_min": 120, "ideal_knee_max": 160, "ideal_trunk": 90},
    "toe_loop": {"type": "jump", "base_value": 0.40, "ideal_knee_min": 130, "ideal_knee_max": 170, "ideal_trunk": 90},
    "salchow": {"type": "jump", "base_value": 0.40, "ideal_knee_min": 125, "ideal_knee_max": 168, "ideal_trunk": 90},
    "camel_spin": {"type": "spin", "base_value": 1.20, "ideal_knee_min": 145, "ideal_knee_max": 180, "ideal_trunk": 40},
    "sit_spin": {"type": "spin", "base_value": 1.20, "ideal_knee_min": 70, "ideal_knee_max": 110, "ideal_trunk": 70},
    "layback_spin": {"type": "spin", "base_value": 1.20, "ideal_knee_min": 145, "ideal_knee_max": 180, "ideal_trunk": 115},
    "step_sequence": {"type": "steps", "base_value": 1.80, "ideal_knee_min": 120, "ideal_knee_max": 165, "ideal_trunk": 90},
}

@st.cache_resource
def init_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_pose, pose


def midpoint(a, b):
    return {"x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2}


def calculate_angle(p1, p2, p3):
    a = np.array([p1["x"], p1["y"]])
    b = np.array([p2["x"], p2["y"]])
    c = np.array([p3["x"], p3["y"]])
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def extract_landmarks(results, shape):
    if not results.pose_landmarks:
        return None
    h, w = shape[:2]
    lm = results.pose_landmarks.landmark
    P = mp.solutions.pose.PoseLandmark
    keys = {
        "left_shoulder": P.LEFT_SHOULDER,
        "right_shoulder": P.RIGHT_SHOULDER,
        "left_hip": P.LEFT_HIP,
        "right_hip": P.RIGHT_HIP,
        "left_knee": P.LEFT_KNEE,
        "right_knee": P.RIGHT_KNEE,
        "left_ankle": P.LEFT_ANKLE,
        "right_ankle": P.RIGHT_ANKLE,
    }
    out = {}
    for name, idx in keys.items():
        l = lm[idx.value]
        out[name] = {"x": l.x * w, "y": l.y * h, "v": l.visibility}
    return out


def analyze_angles(landmarks):
    if not landmarks:
        return {}
    left_knee = calculate_angle(landmarks["left_hip"], landmarks["left_knee"], landmarks["left_ankle"])
    right_knee = calculate_angle(landmarks["right_hip"], landmarks["right_knee"], landmarks["right_ankle"])
    sh_mid = midpoint(landmarks["left_shoulder"], landmarks["right_shoulder"])
    hip_mid = midpoint(landmarks["left_hip"], landmarks["right_hip"])
    virtual_down = {"x": hip_mid["x"], "y": hip_mid["y"] + 100}
    trunk = calculate_angle(sh_mid, hip_mid, virtual_down)
    return {
        "left_knee": left_knee,
        "right_knee": right_knee,
        "avg_knee": (left_knee + right_knee) / 2,
        "trunk": trunk,
    }


def classify_errors(move_type, angles):
    std = MOVEMENT_STANDARDS[move_type]
    errors = {"critical": [], "major": [], "minor": [], "good": []}
    knee = angles.get("avg_knee", 0)
    trunk = angles.get("trunk", 0)

    if knee < std["ideal_knee_min"] - 25 or knee > std["ideal_knee_max"] + 25:
        errors["critical"].append("زاوية الركبة بعيدة جداً عن النطاق المطلوب")
    elif knee < std["ideal_knee_min"] - 10 or knee > std["ideal_knee_max"] + 10:
        errors["major"].append("زاوية الركبة تحتاج تصحيح واضح")
    elif knee < std["ideal_knee_min"] or knee > std["ideal_knee_max"]:
        errors["minor"].append("زاوية الركبة مقبولة لكن يمكن تحسينها")
    else:
        errors["good"].append("وضع الركبة قريب من المعيار")

    if abs(trunk - std["ideal_trunk"]) > 35:
        errors["critical"].append("ميل الجذع غير مناسب للحركة")
    elif abs(trunk - std["ideal_trunk"]) > 20:
        errors["major"].append("الجذع يحتاج استقامة أو ضبط أفضل")
    elif abs(trunk - std["ideal_trunk"]) > 10:
        errors["minor"].append("يمكن تحسين زاوية الجذع")
    else:
        errors["good"].append("الجذع قريب من الوضع المثالي")

    return errors


def calculate_scores(move_type, errors):
    base = MOVEMENT_STANDARDS[move_type]["base_value"]
    deductions = len(errors["critical"]) * 2 + len(errors["major"]) * 1 + len(errors["minor"]) * 0.5
    goe = max(-5, min(5, 5 - deductions))
    technique = max(0, min(100, 95 - deductions * 12))
    final = round(base + goe * 0.1, 2)
    return {"base_value": base, "goe": goe, "technique": technique, "final_element_score": final}


def draw_overlay(frame, results, errors, angles):
    if not results.pose_landmarks:
        return frame
    mp.solutions.drawing_utils.draw_landmarks(
        frame,
        results.pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
        mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
    )
    color = (0, 255, 0)
    label = "Excellent"
    if errors["critical"]:
        color = (0, 0, 255)
        label = "Critical"
    elif errors["major"]:
        color = (0, 140, 255)
        label = "Major"
    elif errors["minor"]:
        color = (0, 215, 255)
        label = "Minor"
    cv2.rectangle(frame, (10, 10), (370, 120), (15, 23, 42), -1)
    cv2.putText(frame, f"Status: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Knee: {angles.get('avg_knee',0):.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Trunk: {angles.get('trunk',0):.1f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return frame


def generate_reports(scores, errors, angle_series):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "scores": scores,
        "errors": errors,
        "frames_analyzed": len(angle_series),
    }
    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        pd.DataFrame([scores]).to_excel(writer, sheet_name="scores", index=False)
        rows = []
        for level, msgs in errors.items():
            for m in msgs:
                rows.append({"level": level, "message": m})
        pd.DataFrame(rows if rows else [{"level":"good","message":"لا توجد أخطاء مسجلة"}]).to_excel(writer, sheet_name="errors", index=False)
        pd.DataFrame(angle_series).to_excel(writer, sheet_name="angles", index=False)
    excel_buffer.seek(0)

    text = f"""
Pro Analyzer Advanced Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Base Value: {scores['base_value']}
GOE: {scores['goe']}
Technique Score: {scores['technique']}
Final Element Score: {scores['final_element_score']}
""".strip().encode("utf-8")
    return json_bytes, excel_buffer.getvalue(), text


def plot_angles(angle_series):
    if not angle_series:
        return None
    df = pd.DataFrame(angle_series)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["avg_knee"], label="Avg Knee")
    ax.plot(df.index, df["trunk"], label="Trunk")
    ax.set_title("Biomechanical Trend")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig


def main():
    mp_pose, pose = init_pose()

    st.title("⛸️ Pro Analyzer Advanced")
    st.caption("نسخة تشغيلية أولى لتحليل الأداء الفني، تقدير الجودة، وإخراج تقرير قابل للمراجعة.")

    with st.sidebar:
        st.header("الإعدادات")
        move_type = st.selectbox("نوع الحركة", list(MOVEMENT_STANDARDS.keys()))
        st.markdown("### تعليمات")
        st.info("1) اختر الحركة. 2) ارفع فيديو واضح. 3) ابدأ التحليل. 4) راجع الدرجات والأخطاء والتقرير.")
        st.markdown("### معايير أولية")
        std = MOVEMENT_STANDARDS[move_type]
        st.write(f"Base Value: {std['base_value']}")
        st.write(f"Ideal Knee: {std['ideal_knee_min']}–{std['ideal_knee_max']}")
        st.write(f"Ideal Trunk: {std['ideal_trunk']}")

    tab1, tab2, tab3 = st.tabs(["📹 تحليل فيديو", "📊 النتائج", "📘 الحكم والتقارير"])

    if "analysis" not in st.session_state:
        st.session_state.analysis = None

    with tab1:
        st.info("ارفع فيديو يحتوي على اللاعب بشكل واضح وجسم كامل قدر الإمكان للحصول على نتائج أفضل.")
        uploaded = st.file_uploader("ارفع ملف فيديو", type=["mp4", "mov", "avi", "mkv"])
        run = st.button("🚀 بدء التحليل", type="primary", use_container_width=True)

        if uploaded and run:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
            tfile.write(uploaded.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            frame_no = 0
            progress = st.progress(0)
            preview = st.empty()
            angle_series = []
            last_errors = {"critical": [], "major": [], "minor": [], "good": []}

            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_no += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                landmarks = extract_landmarks(results, frame.shape)
                if landmarks:
                    angles = analyze_angles(landmarks)
                    angle_series.append(angles)
                    last_errors = classify_errors(move_type, angles)
                    frame = draw_overlay(frame, results, last_errors, angles)
                if frame_no % 5 == 0:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_no}", use_container_width=True)
                progress.progress(min(frame_no / total, 1.0))

            cap.release()
            os.unlink(tfile.name)

            if angle_series:
                avg_angles = pd.DataFrame(angle_series).mean().to_dict()
                errors = classify_errors(move_type, avg_angles)
                scores = calculate_scores(move_type, errors)
                st.session_state.analysis = {
                    "move_type": move_type,
                    "avg_angles": avg_angles,
                    "errors": errors,
                    "scores": scores,
                    "angle_series": angle_series,
                }
                st.success("اكتمل التحليل بنجاح.")
            else:
                st.error("لم يتم اكتشاف هيكل عظمي واضح في الفيديو.")

    with tab2:
        data = st.session_state.analysis
        if not data:
            st.warning("لا توجد نتائج بعد. ابدأ بتحليل فيديو أولاً.")
        else:
            scores = data["scores"]
            avg = data["avg_angles"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Base Value", scores["base_value"])
            c2.metric("GOE", scores["goe"])
            c3.metric("Technique", f"{scores['technique']:.1f}/100")
            c4.metric("Element Score", scores["final_element_score"])

            st.markdown("### المتوسطات البيوميكانيكية")
            a1, a2 = st.columns(2)
            a1.markdown(f"<div class='metric-card'>متوسط زاوية الركبة<br><span class='small-note'>{avg['avg_knee']:.1f}°</span></div>", unsafe_allow_html=True)
            a2.markdown(f"<div class='metric-card'>متوسط زاوية الجذع<br><span class='small-note'>{avg['trunk']:.1f}°</span></div>", unsafe_allow_html=True)

            st.markdown("### ملخص الأخطاء")
            errs = data["errors"]
            if errs["critical"]:
                for e in errs["critical"]:
                    st.error(f"🔴 {e}")
            if errs["major"]:
                for e in errs["major"]:
                    st.warning(f"🟠 {e}")
            if errs["minor"]:
                for e in errs["minor"]:
                    st.info(f"🟡 {e}")
            if errs["good"] and not (errs["critical"] or errs["major"]):
                for e in errs["good"]:
                    st.success(f"✅ {e}")

            fig = plot_angles(data["angle_series"])
            if fig:
                st.pyplot(fig, use_container_width=True)

    with tab3:
        data = st.session_state.analysis
        if not data:
            st.warning("التقارير ستظهر بعد أول تحليل ناجح.")
        else:
            scores = data["scores"]
            errors = data["errors"]
            json_bytes, excel_bytes, text_bytes = generate_reports(scores, errors, data["angle_series"])

            st.markdown("### لوحة الحكم المبسطة")
            table = pd.DataFrame([
                {"البند": "عنصر الحركة", "القيمة": data['move_type']},
                {"البند": "Base Value", "القيمة": scores['base_value']},
                {"البند": "GOE", "القيمة": scores['goe']},
                {"البند": "Technique Score", "القيمة": scores['technique']},
                {"البند": "Final Element Score", "القيمة": scores['final_element_score']},
            ])
            st.dataframe(table, use_container_width=True, hide_index=True)

            st.markdown("### تنزيل التقارير")
            d1, d2, d3 = st.columns(3)
            d1.download_button("تحميل JSON", data=json_bytes, file_name="analysis_report.json", mime="application/json", use_container_width=True)
            d2.download_button("تحميل Excel", data=excel_bytes, file_name="analysis_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            d3.download_button("تحميل TXT", data=text_bytes, file_name="analysis_report.txt", mime="text/plain", use_container_width=True)

            st.markdown("### توصيات أولية")
            if errors["critical"]:
                st.error("الأولوية الآن هي تصحيح الركبة أو الجذع قبل الانتقال إلى تحسينات الأداء الدقيقة.")
            elif errors["major"]:
                st.warning("الأداء قابل للتطوير بسرعة عبر تثبيت الجذع وتحسين ميكانيكية الثني والامتداد.")
            else:
                st.success("الأساس الحركي جيد، والخطوة التالية هي تحسين الجودة ورفع GOE.")

if __name__ == "__main__":
    main()

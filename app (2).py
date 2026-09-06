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
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
[data-testid="stMetricValue"] {font-size: 1.6rem;}
.panel {
  background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
  border: 1px solid #dbeafe;
  border-radius: 18px;
  padding: 1rem;
  box-shadow: 0 6px 16px rgba(15,23,42,0.05);
}
.phase-box {
  border-radius: 14px;
  padding: .75rem .9rem;
  border: 1px solid #e2e8f0;
  background: #ffffff;
}
.small {font-size:0.92rem;color:#475569;}
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
        "hip_y": hip_mid["y"],
        "shoulder_y": sh_mid["y"],
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


def detect_phases(angle_series):
    if not angle_series:
        return {}
    df = pd.DataFrame(angle_series)
    n = len(df)
    prep_end = max(1, int(n * 0.30))
    flight_end = max(prep_end + 1, int(n * 0.65))
    landing_start = max(flight_end, int(n * 0.80))
    phases = {
        "preparation": df.iloc[:prep_end].mean(numeric_only=True).to_dict(),
        "takeoff_flight": df.iloc[prep_end:flight_end].mean(numeric_only=True).to_dict(),
        "landing": df.iloc[landing_start:].mean(numeric_only=True).to_dict(),
    }
    return phases


def score_phase(move_type, phase_name, phase_values):
    std = MOVEMENT_STANDARDS[move_type]
    knee = phase_values.get("avg_knee", 0)
    trunk = phase_values.get("trunk", 0)
    knee_mid = (std["ideal_knee_min"] + std["ideal_knee_max"]) / 2
    knee_penalty = min(40, abs(knee - knee_mid) * 0.7)
    trunk_penalty = min(35, abs(trunk - std["ideal_trunk"]) * 0.9)
    raw = max(0, 100 - knee_penalty - trunk_penalty)
    weights = {"preparation": 0.30, "takeoff_flight": 0.40, "landing": 0.30}
    weighted = raw * weights[phase_name]
    return {"raw": round(raw, 2), "weighted": round(weighted, 2)}


def calculate_scores(move_type, errors, phases):
    base = MOVEMENT_STANDARDS[move_type]["base_value"]
    deductions = len(errors["critical"]) * 2 + len(errors["major"]) * 1 + len(errors["minor"]) * 0.5
    goe = max(-5, min(5, 5 - deductions))

    phase_scores = {name: score_phase(move_type, name, values) for name, values in phases.items()}
    technique = round(sum(v["weighted"] for v in phase_scores.values()), 2)
    final_element = round(base + goe * 0.1, 2)

    pcs = {
        "Skating Skills": round(max(0.25, min(10.0, 4.5 + technique / 20)), 2),
        "Presentation": round(max(0.25, min(10.0, 4.0 + technique / 22)), 2),
        "Composition": round(max(0.25, min(10.0, 4.2 + technique / 21)), 2),
    }
    pcs_total = round(sum(pcs.values()), 2)
    total_segment = round(final_element + pcs_total, 2)
    return {
        "base_value": base,
        "goe": goe,
        "technique": technique,
        "phase_scores": phase_scores,
        "pcs": pcs,
        "pcs_total": pcs_total,
        "final_element_score": final_element,
        "total_segment_estimate": total_segment,
    }


def draw_overlay(frame, results, errors, angles, phase_label):
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
    cv2.rectangle(frame, (10, 10), (420, 145), (15, 23, 42), -1)
    cv2.putText(frame, f"Status: {label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Phase: {phase_label}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Knee: {angles.get('avg_knee',0):.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Trunk: {angles.get('trunk',0):.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return frame


def generate_reports(data):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "move_type": data["move_type"],
        "scores": data["scores"],
        "errors": data["errors"],
        "phases": data["phases"],
        "frames_analyzed": len(data["angle_series"]),
    }
    json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        pd.DataFrame([data["scores"]]).drop(columns=["phase_scores", "pcs"], errors="ignore").to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(data["scores"]["phase_scores"]).T.reset_index().rename(columns={"index":"phase"}).to_excel(writer, sheet_name="phase_scores", index=False)
        pd.DataFrame(list(data["scores"]["pcs"].items()), columns=["component","score"]).to_excel(writer, sheet_name="pcs", index=False)
        rows = []
        for level, msgs in data["errors"].items():
            for m in msgs:
                rows.append({"level": level, "message": m})
        pd.DataFrame(rows if rows else [{"level":"good","message":"لا توجد أخطاء مسجلة"}]).to_excel(writer, sheet_name="errors", index=False)
        pd.DataFrame(data["angle_series"]).to_excel(writer, sheet_name="angles", index=False)
    excel_buffer.seek(0)

    text = f"""
Pro Analyzer Advanced Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Movement: {data['move_type']}
Base Value: {data['scores']['base_value']}
GOE: {data['scores']['goe']}
Technique Score: {data['scores']['technique']}
PCS Total: {data['scores']['pcs_total']}
Estimated Segment Total: {data['scores']['total_segment_estimate']}
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


def recommendations(errors, scores):
    notes = []
    if errors["critical"]:
        notes.append("ابدأ بتصحيح الوضعية الأساسية قبل البحث عن رفع الدرجة، خاصة في ثبات الجذع وثني الركبة.")
    if errors["major"]:
        notes.append("ركّز على الانتقال السلس بين التحضير والهبوط لتحسين GOE وتقليل الخصومات.")
    if scores["pcs_total"] < 18:
        notes.append("النسخة الحالية توصي بالعمل على جودة العرض العام والاتزان لإرفع PCS التقديري.")
    if not notes:
        notes.append("الأداء الأساسي جيد، والخطوة التالية هي زيادة الثبات وتحسين الجودة الدقيقة للحركة.")
    return notes


def main():
    _, pose = init_pose()

    st.title("⛸️ Pro Analyzer Advanced")
    st.caption("نسخة تشغيلية مطورة: تحليل مراحل الحركة، تقدير GOE، لوحة PCS، وتقارير تدريبية قابلة للمراجعة.")

    with st.sidebar:
        st.header("الإعدادات")
        move_type = st.selectbox("نوع الحركة", list(MOVEMENT_STANDARDS.keys()))
        std = MOVEMENT_STANDARDS[move_type]
        st.info("اختر العنصر، ثم ارفع فيديو واضح، ثم ابدأ التحليل للحصول على درجات فنية وتقدير أداء أقرب لمنطق الحكم.")
        st.markdown("### مرجع سريع")
        st.write(f"نوع العنصر: {std['type']}")
        st.write(f"Base Value: {std['base_value']}")
        st.write(f"Ideal Knee: {std['ideal_knee_min']}–{std['ideal_knee_max']}")
        st.write(f"Ideal Trunk: {std['ideal_trunk']}")

    tab1, tab2, tab3, tab4 = st.tabs(["📹 التحليل", "📊 النتائج", "🧑‍⚖️ الحكم", "📄 التقارير"])
    if "analysis" not in st.session_state:
        st.session_state.analysis = None

    with tab1:
        st.info("لأفضل نتيجة: تصوير جانبي أو مائل، جسم كامل داخل الإطار، وخلفية غير مزدحمة.")
        uploaded = st.file_uploader("ارفع ملف فيديو", type=["mp4", "mov", "avi", "mkv"])
        run = st.button("🚀 بدء التحليل المتقدم", type="primary", use_container_width=True)

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
                    tmp_phase = "Preparation" if frame_no / total < 0.3 else ("Takeoff/Flight" if frame_no / total < 0.8 else "Landing")
                    tmp_errors = classify_errors(move_type, angles)
                    frame = draw_overlay(frame, results, tmp_errors, angles, tmp_phase)
                if frame_no % 5 == 0:
                    preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_no}", use_container_width=True)
                progress.progress(min(frame_no / total, 1.0))

            cap.release()
            os.unlink(tfile.name)

            if angle_series:
                avg_angles = pd.DataFrame(angle_series).mean().to_dict()
                phases = detect_phases(angle_series)
                errors = classify_errors(move_type, avg_angles)
                scores = calculate_scores(move_type, errors, phases)
                st.session_state.analysis = {
                    "move_type": move_type,
                    "avg_angles": avg_angles,
                    "phases": phases,
                    "errors": errors,
                    "scores": scores,
                    "angle_series": angle_series,
                }
                st.success("اكتمل التحليل المتقدم بنجاح.")
            else:
                st.error("لم يتم اكتشاف جسم واضح في الفيديو. جرّب فيديو أوضح أو زاوية تصوير أفضل.")

    with tab2:
        data = st.session_state.analysis
        if not data:
            st.warning("لا توجد نتائج بعد. ابدأ بتحليل فيديو أولاً.")
        else:
            scores = data["scores"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Base Value", scores["base_value"])
            c2.metric("GOE", scores["goe"])
            c3.metric("Technique", f"{scores['technique']:.1f}/100")
            c4.metric("PCS Total", scores["pcs_total"])
            c5.metric("Segment Est.", scores["total_segment_estimate"])

            st.markdown("### مراحل الحركة")
            p1, p2, p3 = st.columns(3)
            for box, key, title in [(p1, "preparation", "Preparation"), (p2, "takeoff_flight", "Takeoff / Flight"), (p3, "landing", "Landing")]:
                vals = data["phases"][key]
                raw = scores["phase_scores"][key]["raw"]
                with box:
                    st.markdown(f"<div class='phase-box'><b>{title}</b><br><span class='small'>Knee: {vals.get('avg_knee',0):.1f}°<br>Trunk: {vals.get('trunk',0):.1f}°<br>Score: {raw:.1f}/100</span></div>", unsafe_allow_html=True)

            st.markdown("### الأخطاء المكتشفة")
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
            st.warning("لوحة الحكم ستظهر بعد أول تحليل ناجح.")
        else:
            s = data["scores"]
            st.markdown("### لوحة الحكم المبسطة")
            judge_df = pd.DataFrame([
                {"البند": "Element", "القيمة": data['move_type']},
                {"البند": "Base Value", "القيمة": s['base_value']},
                {"البند": "GOE (-5 to +5)", "القيمة": s['goe']},
                {"البند": "Technique Score", "القيمة": s['technique']},
                {"البند": "PCS Total", "القيمة": s['pcs_total']},
                {"البند": "Estimated Segment Total", "القيمة": s['total_segment_estimate']},
            ])
            st.dataframe(judge_df, use_container_width=True, hide_index=True)

            st.markdown("### Program Components")
            pcs_df = pd.DataFrame(list(s["pcs"].items()), columns=["Component", "Score"])
            st.dataframe(pcs_df, use_container_width=True, hide_index=True)

            st.markdown("### توصيات المدرب")
            for rec in recommendations(data["errors"], s):
                st.write(f"- {rec}")

    with tab4:
        data = st.session_state.analysis
        if not data:
            st.warning("التقارير ستظهر بعد التحليل.")
        else:
            json_bytes, excel_bytes, text_bytes = generate_reports(data)
            d1, d2, d3 = st.columns(3)
            d1.download_button("تحميل JSON", data=json_bytes, file_name="analysis_report.json", mime="application/json", use_container_width=True)
            d2.download_button("تحميل Excel", data=excel_bytes, file_name="analysis_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            d3.download_button("تحميل TXT", data=text_bytes, file_name="analysis_report.txt", mime="text/plain", use_container_width=True)
            st.success("التقارير جاهزة للتنزيل والمراجعة.")

if __name__ == "__main__":
    main()

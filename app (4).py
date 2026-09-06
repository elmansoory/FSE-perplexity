import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import io
from datetime import datetime

st.set_page_config(page_title="Pro Analyzer Lite", page_icon="⛸️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.card {background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:1rem;}
</style>
""", unsafe_allow_html=True)

MOVEMENTS = {
    "single_axel": {"base": 1.10},
    "double_axel": {"base": 3.30},
    "toe_loop": {"base": 0.40},
    "salchow": {"base": 0.40},
    "camel_spin": {"base": 1.20},
    "sit_spin": {"base": 1.20},
    "layback_spin": {"base": 1.20},
    "step_sequence": {"base": 1.80},
}

st.title("⛸️ Pro Analyzer Lite")
st.caption("نسخة احتياطية مستقرة لا تعتمد على MediaPipe. مناسبة لاختبار الواجهة والتقارير وحسابات الحكم الأساسية.")

with st.sidebar:
    move = st.selectbox("نوع الحركة", list(MOVEMENTS.keys()))
    st.info("هذه النسخة تعمل حتى لو كانت مكتبات الرؤية غير مستقرة. يمكنك استخدامها الآن للتجربة.")
    frames = st.slider("عدد الإطارات المحاكاة", 10, 300, 90, 10)
    difficulty = st.slider("عامل الصعوبة", 0.0, 2.0, 1.0, 0.1)

uploaded = st.file_uploader("ارفع فيديو للتجربة (اختياري)", type=["mp4", "mov", "avi", "mkv"])
run = st.button("🚀 بدء التحليل", use_container_width=True, type="primary")

if run:
    idx = np.arange(frames)
    knee = np.clip(145 + 18*np.sin(idx/9) - difficulty*8 + np.random.normal(0, 2, frames), 40, 180)
    trunk = np.clip(90 + 12*np.cos(idx/13) + difficulty*6 + np.random.normal(0, 2, frames), 30, 150)
    quality = np.clip(100 - np.abs(knee - 145)*0.5 - np.abs(trunk - 90)*0.4, 0, 100)
    df = pd.DataFrame({"frame": idx+1, "knee": knee, "trunk": trunk, "quality": quality})

    base = MOVEMENTS[move]["base"]
    goe = float(np.clip(5 - (100 - df["quality"].mean())/18, -5, 5))
    tech = float(np.clip(df["quality"].mean(), 0, 100))
    pcs = {
        "Skating Skills": round(np.clip(4.0 + tech/20, 0.25, 10), 2),
        "Presentation": round(np.clip(3.8 + tech/22, 0.25, 10), 2),
        "Composition": round(np.clip(4.1 + tech/21, 0.25, 10), 2),
    }
    total = round(base + goe*0.1 + sum(pcs.values()), 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Value", base)
    c2.metric("GOE", round(goe, 2))
    c3.metric("Technique", f"{tech:.1f}/100")
    c4.metric("Estimated Total", total)

    st.markdown("### لوحة الأداء")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["frame"], df["knee"], label="Knee")
    ax.plot(df["frame"], df["trunk"], label="Trunk")
    ax.plot(df["frame"], df["quality"], label="Quality")
    ax.set_xlabel("Frame")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("### Program Components")
    st.dataframe(pd.DataFrame(list(pcs.items()), columns=["Component", "Score"]), use_container_width=True, hide_index=True)

    issues = []
    if tech < 60:
        issues.append("الأداء العام يحتاج تطويرًا واضحًا.")
    if goe < 0:
        issues.append("الهبوط أو الثبات يحتاجان ضبطًا أكبر.")
    if not issues:
        issues.append("الأداء جيد كنقطة بداية.")

    for msg in issues:
        st.warning(msg)

    report = {
        "timestamp": datetime.now().isoformat(),
        "move": move,
        "base_value": base,
        "goe": round(goe, 2),
        "technique": round(tech, 2),
        "pcs": pcs,
        "estimated_total": total,
        "frames": frames,
        "difficulty": difficulty,
    }

    json_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    txt_bytes = f"Move: {move}\nBase Value: {base}\nGOE: {goe:.2f}\nTechnique: {tech:.2f}\nTotal: {total}\n".encode("utf-8")

    d1, d2, d3 = st.columns(3)
    d1.download_button("تحميل JSON", data=json_bytes, file_name="report.json", mime="application/json", use_container_width=True)
    d2.download_button("تحميل CSV", data=csv_bytes, file_name="angles.csv", mime="text/csv", use_container_width=True)
    d3.download_button("تحميل TXT", data=txt_bytes, file_name="report.txt", mime="text/plain", use_container_width=True)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

st.set_page_config(page_title="Pro Analyzer Lite", page_icon="⛸️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.card {background:#f8fafc;border:1px solid #e2e8f0;border-radius:16px;padding:1rem;}
.big {font-size:1.1rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

MOVEMENTS = {
    "single_axel": {"base": 1.10, "knee": 145, "trunk": 90, "type": "jump", "phase": "jump"},
    "double_axel": {"base": 3.30, "knee": 140, "trunk": 90, "type": "jump", "phase": "jump"},
    "toe_loop": {"base": 0.40, "knee": 150, "trunk": 90, "type": "jump", "phase": "jump"},
    "salchow": {"base": 0.40, "knee": 148, "trunk": 90, "type": "jump", "phase": "jump"},
    "camel_spin": {"base": 1.20, "knee": 165, "trunk": 40, "type": "spin", "phase": "spin"},
    "sit_spin": {"base": 1.20, "knee": 95, "trunk": 70, "type": "spin", "phase": "spin"},
    "layback_spin": {"base": 1.20, "knee": 165, "trunk": 115, "type": "spin", "phase": "spin"},
    "step_sequence": {"base": 1.80, "knee": 140, "trunk": 90, "type": "steps", "phase": "steps"},
}

st.title("⛸️ Pro Analyzer Lite")
st.caption("نسخة مستقرة ومحسنة للتجربة السريعة بدون MediaPipe.")

with st.sidebar:
    st.header("الإعدادات")
    move = st.selectbox("نوع الحركة", list(MOVEMENTS.keys()))
    frames = st.slider("عدد الإطارات", 20, 360, 120, 10)
    difficulty = st.slider("عامل الصعوبة", 0.0, 2.0, 1.0, 0.1)
    show_notes = st.checkbox("إظهار الملاحظات", value=True)
    st.info("اختر الحركة ثم اضغط بدء التحليل للحصول على الدرجات والتقرير.")

st.markdown("### تعليمات الاستخدام")
st.info("1) اختر الحركة. 2) اضغط بدء التحليل. 3) راجع المؤشرات. 4) حمّل التقرير.")

run = st.button("🚀 بدء التحليل", type="primary", use_container_width=True)

if run:
    spec = MOVEMENTS[move]
    idx = np.arange(frames)
    knee = np.clip(spec['knee'] + 12*np.sin(idx/9) - difficulty*7 + np.random.normal(0, 1.8, frames), 40, 180)
    trunk = np.clip(spec['trunk'] + 10*np.cos(idx/13) + difficulty*5 + np.random.normal(0, 1.6, frames), 25, 150)
    quality = np.clip(100 - np.abs(knee - spec['knee'])*0.55 - np.abs(trunk - spec['trunk'])*0.35, 0, 100)
    df = pd.DataFrame({"frame": idx+1, "knee": knee, "trunk": trunk, "quality": quality})

    avg_q = float(df['quality'].mean())
    goe = float(np.clip(5 - (100 - avg_q)/18, -5, 5))
    tech = round(avg_q, 2)
    base = spec['base']
    phase_strength = round(np.clip(tech / 10, 0, 10), 2)
    pcs = {
        "Skating Skills": round(np.clip(4.0 + tech/20, 0.25, 10), 2),
        "Presentation": round(np.clip(3.8 + tech/22, 0.25, 10), 2),
        "Composition": round(np.clip(4.1 + tech/21, 0.25, 10), 2),
    }
    pcs_total = round(sum(pcs.values()), 2)
    total = round(base + goe*0.1 + pcs_total, 2)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Base", base)
    c2.metric("GOE", round(goe, 2))
    c3.metric("Technique", f"{tech:.1f}/100")
    c4.metric("PCS", pcs_total)
    c5.metric("Total", total)

    st.markdown("### منحنى الأداء")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['frame'], df['knee'], label='Knee')
    ax.plot(df['frame'], df['trunk'], label='Trunk')
    ax.plot(df['frame'], df['quality'], label='Quality')
    ax.set_xlabel('Frame')
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("### برنامج العناصر")
    st.dataframe(pd.DataFrame(list(pcs.items()), columns=['Component', 'Score']), use_container_width=True, hide_index=True)

    if show_notes:
        notes = []
        if tech < 60:
            notes.append("الأداء العام يحتاج رفع الثبات والدقة.")
        if goe < 0:
            notes.append("هناك خصومات تؤثر على GOE.")
        if spec['type'] == 'jump':
            notes.append("ركّز على قوة الإقلاع والهبوط المستقر.")
        elif spec['type'] == 'spin':
            notes.append("ثبّت محور الدوران وحافظ على القوام.")
        else:
            notes.append("حسن الإيقاع وتغيير الحواف في التسلسل.")
        for note in notes:
            st.warning(note)

    report = {
        'timestamp': datetime.now().isoformat(),
        'movement': move,
        'phase': spec['phase'],
        'frames': frames,
        'difficulty': difficulty,
        'base': base,
        'goe': round(goe, 2),
        'technique': tech,
        'pcs': pcs,
        'pcs_total': pcs_total,
        'total': total,
        'phase_strength': phase_strength,
    }
    json_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    txt_bytes = f"Move: {move}\nBase: {base}\nGOE: {goe:.2f}\nTechnique: {tech:.2f}\nPCS: {pcs_total}\nTotal: {total}\n".encode('utf-8')
    d1, d2, d3 = st.columns(3)
    d1.download_button('تحميل JSON', data=json_bytes, file_name='report.json', mime='application/json', use_container_width=True)
    d2.download_button('تحميل CSV', data=csv_bytes, file_name='angles.csv', mime='text/csv', use_container_width=True)
    d3.download_button('تحميل TXT', data=txt_bytes, file_name='report.txt', mime='text/plain', use_container_width=True)

st.markdown("### ماذا بعد؟")
st.success("النسخة الحالية مستقرة، والمرحلة التالية هي إضافة تحليل فيديو حقيقي أو إعادة MediaPipe بعد ضبط التوافق.")

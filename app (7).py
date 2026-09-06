import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

st.set_page_config(page_title="Pro Analyzer Hybrid", page_icon="⛸️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.panel {background:#f8fafc;border:1px solid #e2e8f0;border-radius:18px;padding:1rem;box-shadow:0 4px 14px rgba(15,23,42,.05);}
.good {color:#15803d;font-weight:700;}
.mid {color:#b45309;font-weight:700;}
.bad {color:#b91c1c;font-weight:700;}
</style>
""", unsafe_allow_html=True)

MOVEMENTS = {
    "single_axel": {"base": 1.10, "knee": 145, "trunk": 90, "type": "jump"},
    "double_axel": {"base": 3.30, "knee": 140, "trunk": 90, "type": "jump"},
    "toe_loop": {"base": 0.40, "knee": 150, "trunk": 90, "type": "jump"},
    "salchow": {"base": 0.40, "knee": 148, "trunk": 90, "type": "jump"},
    "camel_spin": {"base": 1.20, "knee": 165, "trunk": 40, "type": "spin"},
    "sit_spin": {"base": 1.20, "knee": 95, "trunk": 70, "type": "spin"},
    "layback_spin": {"base": 1.20, "knee": 165, "trunk": 115, "type": "spin"},
    "step_sequence": {"base": 1.80, "knee": 140, "trunk": 90, "type": "steps"},
}

def simulate_analysis(move, frames, difficulty):
    spec = MOVEMENTS[move]
    idx = np.arange(frames)
    knee = np.clip(spec['knee'] + 12*np.sin(idx/9) - difficulty*7 + np.random.normal(0, 1.8, frames), 40, 180)
    trunk = np.clip(spec['trunk'] + 10*np.cos(idx/13) + difficulty*5 + np.random.normal(0, 1.6, frames), 25, 150)
    quality = np.clip(100 - np.abs(knee - spec['knee'])*0.55 - np.abs(trunk - spec['trunk'])*0.35, 0, 100)
    df = pd.DataFrame({'frame': idx+1, 'knee': knee, 'trunk': trunk, 'quality': quality})
    avg_q = float(df['quality'].mean())
    goe = float(np.clip(5 - (100 - avg_q)/18, -5, 5))
    pcs = {
        'Skating Skills': round(np.clip(4.0 + avg_q/20, 0.25, 10), 2),
        'Presentation': round(np.clip(3.8 + avg_q/22, 0.25, 10), 2),
        'Composition': round(np.clip(4.1 + avg_q/21, 0.25, 10), 2),
    }
    total = round(spec['base'] + goe*0.1 + sum(pcs.values()), 2)
    return df, avg_q, goe, pcs, total

st.title("⛸️ Pro Analyzer Hybrid")
st.caption("نسخة هجينة تجمع الاستقرار مع واجهة أكثر احترافية، ومعها وضع Live Demo بدل الكاميرا الحقيقية عند الحاجة.")

with st.sidebar:
    st.header("الإعدادات")
    mode = st.radio("وضع التشغيل", ["Hybrid Demo", "Live Preview"], index=0)
    move = st.selectbox("نوع الحركة", list(MOVEMENTS.keys()))
    frames = st.slider("عدد الإطارات", 20, 360, 120, 10)
    difficulty = st.slider("عامل الصعوبة", 0.0, 2.0, 1.0, 0.1)
    show_notes = st.checkbox("إظهار الملاحظات", value=True)
    st.info("Hybrid Demo يستخدم محاكاة مستقرة. Live Preview يستخدم كاميرا الويب عبر st.camera_input بشكل صورة واحدة.")

st.info("هذه النسخة مستقرة ومناسبة الآن للتجربة، مع فصل واضح بين العرض الحي والتجربة الآمنة.")

if mode == "Live Preview":
    pic = st.camera_input("التقط صورة من الكاميرا")
    if pic is not None:
        st.image(pic, caption="Live Preview Snapshot", use_container_width=True)
        st.success("تم التقاط صورة. هذه النسخة تعرض لقطة مباشرة وتستطيع توسيعها لاحقاً إلى بث فيديو كامل.")

run = st.button("🚀 بدء التحليل", type="primary", use_container_width=True)

if run:
    df, avg_q, goe, pcs, total = simulate_analysis(move, frames, difficulty)
    spec = MOVEMENTS[move]
    base = spec['base']

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Base", base)
    c2.metric("GOE", round(goe, 2))
    c3.metric("Technique", f"{avg_q:.1f}/100")
    c4.metric("PCS", round(sum(pcs.values()), 2))
    c5.metric("Total", total)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['frame'], df['knee'], label='Knee')
    ax.plot(df['frame'], df['trunk'], label='Trunk')
    ax.plot(df['frame'], df['quality'], label='Quality')
    ax.set_xlabel('Frame')
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    st.markdown("### Program Components")
    st.dataframe(pd.DataFrame(list(pcs.items()), columns=['Component', 'Score']), use_container_width=True, hide_index=True)

    if show_notes:
        if avg_q < 60:
            st.markdown('<p class="bad">الأداء يحتاج تطويرًا واضحًا.</p>', unsafe_allow_html=True)
        elif goe < 0:
            st.markdown('<p class="mid">هناك خصومات تؤثر على GOE.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="good">الأداء جيد كبداية.</p>', unsafe_allow_html=True)
        if spec['type'] == 'jump':
            st.info('ركّز على قوة الإقلاع والهبوط المستقر.')
        elif spec['type'] == 'spin':
            st.info('ثبّت محور الدوران وحافظ على القوام.')
        else:
            st.info('حسن الإيقاع وتغيير الحواف في التسلسل.')

    report = {
        'timestamp': datetime.now().isoformat(),
        'movement': move,
        'mode': mode,
        'frames': frames,
        'difficulty': difficulty,
        'base': base,
        'goe': round(goe, 2),
        'technique': round(avg_q, 2),
        'pcs': pcs,
        'total': total,
    }
    json_bytes = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    txt_bytes = f"Move: {move}\nBase: {base}\nGOE: {goe:.2f}\nTechnique: {avg_q:.2f}\nPCS: {round(sum(pcs.values()),2)}\nTotal: {total}\n".encode('utf-8')

    d1, d2, d3 = st.columns(3)
    d1.download_button('تحميل JSON', data=json_bytes, file_name='report.json', mime='application/json', use_container_width=True)
    d2.download_button('تحميل CSV', data=csv_bytes, file_name='analysis.csv', mime='text/csv', use_container_width=True)
    d3.download_button('تحميل TXT', data=txt_bytes, file_name='report.txt', mime='text/plain', use_container_width=True)

st.markdown('### القادم')
st.write('الخطوة التالية: دمج فيديو حقيقي أو تحويل Live Preview إلى بث متكرر مع streamlit-webrtc على Python 3.11/3.12.')

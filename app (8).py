import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

st.set_page_config(page_title="Pro Analyzer Hybrid Pro", page_icon="⛸️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 0.9rem;}
.hero {background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%); padding: 1.2rem 1.2rem; border-radius: 24px; color: white; box-shadow: 0 10px 30px rgba(15,23,42,.15);}
.card {background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;padding:1rem;box-shadow:0 6px 18px rgba(15,23,42,.05);}
.subtle {color:#475569;}
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

st.markdown('<div class="hero"><h1>⛸️ Pro Analyzer Hybrid Pro</h1><p>نسخة احترافية جدًا تجمع بين الاستقرار والواجهة الحديثة وخطة التوسعة المستقبلية.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("لوحة التحكم")
    mode = st.radio("وضع التشغيل", ["Hybrid Demo", "Live Preview"], index=0)
    move = st.selectbox("نوع الحركة", list(MOVEMENTS.keys()))
    frames = st.slider("عدد الإطارات", 20, 360, 140, 10)
    difficulty = st.slider("عامل الصعوبة", 0.0, 2.0, 1.0, 0.1)
    style = st.select_slider("شدة التحليل", options=["Low", "Balanced", "High"], value="Balanced")
    show_notes = st.checkbox("إظهار الملاحظات", value=True)
    st.caption("Live Preview يعرض لقطة من الكاميرا. Full Stream يمكن إضافته لاحقًا عبر streamlit-webrtc.")

st.markdown("### بداية سريعة")
st.info("اختر الوضع ثم الحركة، وبعدها شغّل التحليل للحصول على لوحة نتائج وتقارير قابلة للتنزيل.")

if mode == "Live Preview":
    pic = st.camera_input("التقط صورة مباشرة من الكاميرا")
    if pic is not None:
        st.image(pic, caption="Live Preview Snapshot", use_container_width=True)
        st.success("تم استقبال صورة الكاميرا بنجاح.")


def simulate_analysis(move, frames, difficulty, style):
    spec = MOVEMENTS[move]
    idx = np.arange(frames)
    style_bias = {"Low": 0.6, "Balanced": 1.0, "High": 1.35}[style]
    knee = np.clip(spec['knee'] + 12*np.sin(idx/9) - difficulty*7*style_bias + np.random.normal(0, 1.7, frames), 40, 180)
    trunk = np.clip(spec['trunk'] + 10*np.cos(idx/13) + difficulty*5*style_bias + np.random.normal(0, 1.5, frames), 25, 150)
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

run = st.button("🚀 بدء التحليل", type="primary", use_container_width=True)

if run:
    df, avg_q, goe, pcs, total = simulate_analysis(move, frames, difficulty, style)
    spec = MOVEMENTS[move]
    base = spec['base']
    intensity_label = {"Low": "تحليل خفيف", "Balanced": "تحليل متوازن", "High": "تحليل عالي"}[style]

    cols = st.columns(5)
    cols[0].metric("Base", base)
    cols[1].metric("GOE", round(goe, 2))
    cols[2].metric("Technique", f"{avg_q:.1f}/100")
    cols[3].metric("PCS", round(sum(pcs.values()), 2))
    cols[4].metric("Total", total)

    left, right = st.columns([2,1])
    with left:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['frame'], df['knee'], label='Knee', linewidth=2)
        ax.plot(df['frame'], df['trunk'], label='Trunk', linewidth=2)
        ax.plot(df['frame'], df['quality'], label='Quality', linewidth=2)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    with right:
        st.markdown('<div class="card"><h4>تفاصيل سريعة</h4>', unsafe_allow_html=True)
        st.write(f"Mode: {mode}")
        st.write(f"Intensity: {intensity_label}")
        st.write(f"Movement Type: {spec['type']}")
        st.write(f"Frames: {frames}")
        st.write(f"Difficulty: {difficulty}")
        st.markdown('</div>', unsafe_allow_html=True)

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
        'style': style,
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

st.markdown('### خارطة التطوير التالية')
st.write('1) Live Camera Full Stream عبر streamlit-webrtc. 2) دمج MediaPipe على Python 3.11/3.12. 3) تصميم Dashboard أكثر احترافية مع تبويبات ولوحة أداء.')

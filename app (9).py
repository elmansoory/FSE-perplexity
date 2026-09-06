import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

st.set_page_config(page_title="World Champion Coach Pro v3", page_icon="⛸️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>.block-container{padding-top:.8rem}.hero{background:linear-gradient(135deg,#0f172a,#1d4ed8 55%,#38bdf8);color:#fff;padding:1.3rem;border-radius:24px}.card{background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:1rem}</style>""", unsafe_allow_html=True)

TEXT={"Arabic":{"title":"World Champion Coach Pro v3","subtitle":"منصة تدريب وتحليل للتزلج الفني: إعدادات المدرب، تقييم اللاعب، خطة أسبوعية ومحاكي حكم.","coach_settings":"إعدادات المدرب","dashboard":"لوحة اللاعب","evaluation":"واجهة التقييم","weekly":"الخطة الأسبوعية","judge":"محاكي الحكم","rules":"القواعد والمراجع","coach_name":"اسم المدرب","team_name":"اسم الفريق","athlete":"اسم اللاعب","season_goal":"هدف الموسم","competition":"المنافسة القادمة","discipline":"الفئة","save":"حفظ إعدادات المدرب","saved":"تم حفظ الإعدادات لهذه الجلسة.","skill":"العنصر","phase":"مرحلة التدريب","frames":"حجم العينة","difficulty":"شدة التحدي","precision":"دقة اللاعب","fatigue":"إرهاق اللاعب","run":"🚀 حلل الأداء وابنِ الخطة","base":"القيمة الأساسية","goe":"GOE","execution":"التنفيذ","pcs":"PCS","target":"النتيجة التقديرية","plan":"خطة التدريب","corrections":"أولويات التصحيح","references":"المراجع الرسمية","download_json":"تحميل تقرير JSON","download_csv":"تحميل بيانات CSV","download_weekly":"تحميل الخطة الأسبوعية CSV","weekly_focus":"أولوية الأسبوع"},"English":{"title":"World Champion Coach Pro v3","subtitle":"Figure-skating coaching and analysis platform: coach settings, athlete evaluation, weekly planning, and judge simulation.","coach_settings":"Coach Settings","dashboard":"Athlete Dashboard","evaluation":"Evaluation Interface","weekly":"Weekly Plan","judge":"Judge Simulator","rules":"Rules & References","coach_name":"Coach name","team_name":"Team name","athlete":"Athlete name","season_goal":"Season goal","competition":"Next competition","discipline":"Discipline","save":"Save coach settings","saved":"Settings saved for this session.","skill":"Element","phase":"Training phase","frames":"Sample size","difficulty":"Challenge level","precision":"Athlete precision","fatigue":"Athlete fatigue","run":"🚀 Analyze Performance & Build Plan","base":"Base Value","goe":"GOE","execution":"Execution","pcs":"PCS","target":"Estimated Total","plan":"Training Plan","corrections":"Priority Corrections","references":"Official References","download_json":"Download JSON Report","download_csv":"Download CSV Data","download_weekly":"Download Weekly Plan CSV","weekly_focus":"Weekly priority"}}
MOVES={"single_axel":{"base":1.10,"type":"jump","focus":["takeoff","air position","landing"]},"double_axel":{"base":3.30,"type":"jump","focus":["edge control","rotation","landing"]},"toe_loop":{"base":0.40,"type":"jump","focus":["entry edge","air tightness","exit"]},"salchow":{"base":0.40,"type":"jump","focus":["entry curve","power","flow"]},"camel_spin":{"base":1.20,"type":"spin","focus":["center","position","speed"]},"sit_spin":{"base":1.20,"type":"spin","focus":["depth","axis","speed"]},"layback_spin":{"base":1.20,"type":"spin","focus":["back line","axis","speed"]},"step_sequence":{"base":1.80,"type":"steps","focus":["edges","counters","rhythm"]}}
REFS=[("ISU Judging System","https://www.isu.org/figure-skating/rules/fsk-judging-system"),("ISU Sports Rules","https://www.isu.org/figure-skating-rules/"),("U.S. Figure Skating Rules & Resources","https://www.usfigureskating.org/about/rules")]

def simulate(skill,frames,difficulty,precision,fatigue):
    spec=MOVES[skill]; x=np.arange(frames); kt=145 if spec["type"]=="jump" else 120; tt=65 if spec["type"]=="spin" else 90
    knee=np.clip(kt+14*np.sin(x/8)-difficulty*8-fatigue*.9+np.random.normal(0,1.6,frames),35,180)
    trunk=np.clip(tt+11*np.cos(x/11)+difficulty*5-fatigue*.4+np.random.normal(0,1.4,frames),20,150)
    quality=np.clip(100-np.abs(knee-kt)*.45-np.abs(trunk-tt)*.35+precision*1.5,0,100)
    df=pd.DataFrame({"frame":x+1,"knee":knee,"trunk":trunk,"quality":quality}); execution=float(df.quality.mean()); goe=float(np.clip((execution-70)/6,-5,5))
    pcs={"Skating Skills":round(np.clip(4+execution/20,.25,10),2),"Composition":round(np.clip(4.2+execution/21,.25,10),2),"Performance":round(np.clip(4.1+execution/20.5,.25,10),2),"Interpretation":round(np.clip(4+execution/22,.25,10),2)}
    return df,execution,goe,pcs,round(spec["base"]+goe*.1+sum(pcs.values()),2)

def plan_for(kind,phase):
    data={"jump":{"On-Ice":["Edge drills","Takeoff entries","3 clean jump reps","Landing hold for 2 seconds"],"Off-Ice":["Single-leg strength","Snap-downs","Core anti-rotation","Plyometric control"],"Drills":["Air-position squeeze","Rotation timing","Stick-the-landing drill","Landing alignment"],"Competition Prep":["Run-through timing","Quality reps only","Mental cue routine","Recovery glide"],"Recovery":["Light skating","Mobility","Breath reset","Sleep prioritization"]},"spin":{"On-Ice":["Entry control","Centering","Speed maintenance","Exit balance"],"Off-Ice":["Balance board","Axis hold","Hip-line drills","Shoulder alignment"],"Drills":["Spot checks","Free-leg consistency","Core compression","Position switches"],"Competition Prep":["Short spin sets","Quality centering","Relaxed breathing","Confidence cue"],"Recovery":["Mobility flow","Hip release","Ankle reset","Easy edges"]},"steps":{"On-Ice":["Edge sequence","Turns","Rhythm shift","Speed changes"],"Off-Ice":["Agility ladder","Ankle stiffness","Cadence work","Footwork memory"],"Drills":["Four-corner turns","Count-based sequences","Acceleration patterns","Pattern holds"],"Competition Prep":["Music timing","Clean exits","Step quality under fatigue","Performance intent"],"Recovery":["Easy stroking","Range of motion","Foot massage","Breathing"]}}
    return data[kind][phase]

with st.sidebar:
    lang=st.selectbox("Language / اللغة",["Arabic","English"]); t=TEXT[lang]
st.markdown(f'<div class="hero"><h1>⛸️ {t["title"]}</h1><p>{t["subtitle"]}</p></div>',unsafe_allow_html=True)
tabs=st.tabs([t["coach_settings"],t["dashboard"],t["evaluation"],t["weekly"],t["judge"],t["rules"]])
with tabs[0]:
    a,b=st.columns(2)
    with a:
        coach=st.text_input(t["coach_name"],st.session_state.get("coach","Head Coach")); team=st.text_input(t["team_name"],st.session_state.get("team","World Team")); discipline=st.selectbox(t["discipline"],["Singles","Pairs","Ice Dance","Synchronized Skating"])
    with b:
        athlete=st.text_input(t["athlete"],st.session_state.get("athlete","Skater")); goal=st.text_input(t["season_goal"],st.session_state.get("goal","World Championship Readiness")); competition=st.text_input(t["competition"],st.session_state.get("competition","Next Competition"))
    if st.button(t["save"],type="primary"):
        st.session_state.update({"coach":coach,"team":team,"athlete":athlete,"goal":goal,"competition":competition}); st.success(t["saved"])
with tabs[1]:
    c1,c2,c3,c4=st.columns(4); c1.metric(t["athlete"],st.session_state.get("athlete","Skater")); c2.metric(t["coach_name"],st.session_state.get("coach","Head Coach")); c3.metric(t["team_name"],st.session_state.get("team","World Team")); c4.metric(t["season_goal"],st.session_state.get("goal","World Championship Readiness"))
with tabs[2]:
    a,b,c=st.columns(3)
    with a: skill=st.selectbox(t["skill"],list(MOVES)); phase=st.selectbox(t["phase"],["On-Ice","Off-Ice","Drills","Competition Prep","Recovery"]); focus=st.selectbox(t["weekly_focus"],["Technical","Performance","Consistency","Recovery"])
    with b: frames=st.slider(t["frames"],40,360,120,10); difficulty=st.slider(t["difficulty"],0.,2.,1.,.1)
    with c: precision=st.slider(t["precision"],0.,10.,5.,.5); fatigue=st.slider(t["fatigue"],0.,10.,3.,.5)
    if st.button(t["run"],type="primary",use_container_width=True):
        df,execution,goe,pcs,total=simulate(skill,frames,difficulty,precision,fatigue); spec=MOVES[skill]; issues=[]
        if execution<60: issues.append("Major stability issue / مشكلة كبيرة في الثبات")
        if goe<0: issues.append("Likely GOE deductions / خصومات GOE محتملة")
        if fatigue>6: issues.append("High fatigue: reduce load and add recovery / إرهاق مرتفع")
        if precision<5: issues.append("Precision below target / الدقة أقل من الهدف")
        st.session_state["analysis"]={"df":df,"execution":execution,"goe":goe,"pcs":pcs,"total":total,"skill":skill,"phase":phase,"plan":plan_for(spec["type"],phase),"issues":issues,"base":spec["base"],"focus":spec["focus"],"weekly_focus":focus}
    z=st.session_state.get("analysis")
    if z:
        m=st.columns(5); m[0].metric(t["base"],z["base"]); m[1].metric(t["goe"],round(z["goe"],2)); m[2].metric(t["execution"],f'{z["execution"]:.1f}/100'); m[3].metric(t["pcs"],round(sum(z["pcs"].values()),2)); m[4].metric(t["target"],z["total"])
        fig,ax=plt.subplots(figsize=(10,4)); ax.plot(z["df"].frame,z["df"].knee,label="Knee"); ax.plot(z["df"].frame,z["df"].trunk,label="Trunk"); ax.plot(z["df"].frame,z["df"].quality,label="Quality"); ax.grid(alpha=.25); ax.legend(); st.pyplot(fig,use_container_width=True)
        st.markdown(f'### {t["plan"]}'); [st.success(i) for i in z["plan"]]; st.markdown(f'### {t["corrections"]}'); [st.warning(i) for i in z["issues"] or ["Performance is progressing well / الأداء يتقدم بشكل جيد"]]
with tabs[3]:
    z=st.session_state.get("analysis")
    if not z: st.info("Run an evaluation first / شغّل التقييم أولًا")
    else:
        p=z["plan"]; weekly=pd.DataFrame([["Mon","On-Ice",p[0]],["Tue","Off-Ice",p[1]],["Wed","Drills",p[2]],["Thu","On-Ice",p[0]],["Fri","Competition Prep","Run-through + quality reps"],["Sat","Performance","Full program simulation"],["Sun","Recovery","Mobility + light skating"]],columns=["Day","Phase","Main Work"]); st.session_state["weekly"]=weekly; st.dataframe(weekly,use_container_width=True,hide_index=True); st.download_button(t["download_weekly"],weekly.to_csv(index=False).encode(),"weekly_plan.csv","text/csv")
with tabs[4]:
    z=st.session_state.get("analysis")
    if not z: st.info("Run an evaluation first / شغّل التقييم أولًا")
    else: st.dataframe(pd.DataFrame([["Technical Value",z["base"]],["GOE",round(z["goe"],2)],["Skating Skills",z["pcs"]["Skating Skills"]],["Composition",z["pcs"]["Composition"]],["Performance",z["pcs"]["Performance"]],["Interpretation",z["pcs"]["Interpretation"]]],columns=["Metric","Score"]),use_container_width=True,hide_index=True); st.caption("Training estimate only — not an official ISU score.")
with tabs[5]:
    st.subheader(t["references"]); [st.markdown(f'- [{name}]({url})') for name,url in REFS]
z=st.session_state.get("analysis")
if z:
    weekly=st.session_state.get("weekly"); report={"timestamp":datetime.now().isoformat(),"coach":st.session_state.get("coach","Head Coach"),"team":st.session_state.get("team","World Team"),"athlete":st.session_state.get("athlete","Skater"),"skill":z["skill"],"phase":z["phase"],"base":z["base"],"goe":round(z["goe"],2),"execution":round(z["execution"],2),"pcs":z["pcs"],"total":z["total"],"training_plan":z["plan"],"issues":z["issues"],"weekly_plan":weekly.to_dict(orient="records") if weekly is not None else [],"references":REFS}; st.divider(); st.download_button(t["download_json"],json.dumps(report,ensure_ascii=False,indent=2).encode("utf-8"),"world_champion_coach_report.json","application/json"); st.download_button(t["download_csv"],z["df"].to_csv(index=False).encode(),"analysis_data.csv","text/csv")

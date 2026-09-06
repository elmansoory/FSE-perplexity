import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

st.set_page_config(
    page_title="World Champion Coach Pro v3",
    page_icon="⛸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 0.8rem;}
.hero {background:linear-gradient(135deg,#0f172a 0%,#1d4ed8 55%,#38bdf8 100%);color:white;padding:1.3rem;border-radius:24px;box-shadow:0 12px 32px rgba(15,23,42,.18);}
.card {background:#fff;border:1px solid #e2e8f0;border-radius:18px;padding:1rem;box-shadow:0 6px 18px rgba(15,23,42,.05);}
.small {color:#475569;}
</style>
""", unsafe_allow_html=True)

TEXT = {
    "Arabic": {
        "title": "World Champion Coach Pro v3",
        "subtitle": "منصة تدريب وتحليل للتزلج الفني: إعدادات المدرب، تقييم اللاعب، خطة أسبوعية ومحاكي حكم.",
        "coach_settings": "إعدادات المدرب",
        "dashboard": "لوحة اللاعب",
        "evaluation": "واجهة التقييم",
        "weekly": "الخطة الأسبوعية",
        "judge": "محاكي الحكم",
        "rules": "القواعد والمراجع",
        "coach_name": "اسم المدرب",
        "team_name": "اسم الفريق",
        "athlete": "اسم اللاعب",
        "season_goal": "هدف الموسم",
        "competition": "المنافسة القادمة",
        "discipline": "الفئة",
        "save": "حفظ إعدادات المدرب",
        "saved": "تم حفظ الإعدادات لهذه الجلسة.",
        "skill": "العنصر",
        "phase": "مرحلة التدريب",
        "frames": "حجم العينة",
        "difficulty": "شدة التحدي",
        "precision": "دقة اللاعب",
        "fatigue": "إرهاق اللاعب",
        "run": "🚀 حلل الأداء وابنِ الخطة",
        "base": "القيمة الأساسية",
        "goe": "GOE",
        "execution": "التنفيذ",
        "pcs": "PCS",
        "target": "النتيجة التقديرية",
        "plan": "خطة التدريب",
        "corrections": "أولويات التصحيح",
        "references": "المراجع الرسمية",
        "download_json": "تحميل تقرير JSON",
        "download_csv": "تحميل بيانات CSV",
        "download_weekly": "تحميل الخطة الأسبوعية CSV",
        "language": "لغة الواجهة",
        "weekly_focus": "أولوية الأسبوع",
        "analysis": "تحليل الأداء",
        "settings_note": "هذه الإعدادات تحدد طريقة عرض البرامج التدريبية والتقييم.",
    },
    "English": {
        "title": "World Champion Coach Pro v3",
        "subtitle": "Figure-skating coaching and analysis platform: coach settings, athlete evaluation, weekly planning, and judge simulation.",
        "coach_settings": "Coach Settings",
        "dashboard": "Athlete Dashboard",
        "evaluation": "Evaluation Interface",
        "weekly": "Weekly Plan",
        "judge": "Judge Simulator",
        "rules": "Rules & References",
        "coach_name": "Coach name",
        "team_name": "Team name",
        "athlete": "Athlete name",
        "season_goal": "Season goal",
        "competition": "Next competition",
        "discipline": "Discipline",
        "save": "Save coach settings",
        "saved": "Settings saved for this session.",
        "skill": "Element",
        "phase": "Training phase",
        "frames": "Sample size",
        "difficulty": "Challenge level",
        "precision": "Athlete precision",
        "fatigue": "Athlete fatigue",
        "run": "🚀 Analyze Performance & Build Plan",
        "base": "Base Value",
        "goe": "GOE",
        "execution": "Execution",
        "pcs": "PCS",
        "target": "Estimated Total",
        "plan": "Training Plan",
        "corrections": "Priority Corrections",
        "references": "Official References",
        "download_json": "Download JSON Report",
        "download_csv": "Download CSV Data",
        "download_weekly": "Download Weekly Plan CSV",
        "language": "Interface language",
        "weekly_focus": "Weekly priority",
        "analysis": "Performance Analysis",
        "settings_note": "These settings control how training plans and evaluations are presented.",
    },
}

MOVE_LIBRARY = {
    "single_axel": {"base": 1.10, "type": "jump", "focus": ["takeoff", "air position", "landing"]},
    "double_axel": {"base": 3.30, "type": "jump", "focus": ["edge control", "rotation", "landing"]},
    "toe_loop": {"base": 0.40, "type": "jump", "focus": ["entry edge", "air tightness", "exit"]},
    "salchow": {"base": 0.40, "type": "jump", "focus": ["entry curve", "power", "flow"]},
    "camel_spin": {"base": 1.20, "type": "spin", "focus": ["center", "position", "speed"]},
    "sit_spin": {"base": 1.20, "type": "spin", "focus": ["depth", "axis", "speed"]},
    "layback_spin": {"base": 1.20, "type": "spin", "focus": ["back line", "axis", "speed"]},
    "step_sequence": {"base": 1.80, "type": "steps", "focus": ["edges", "counters", "rhythm"]},
}

REFERENCES = [
    ("ISU Judging System", "https://www.isu.org/figure-skating/rules/fsk-judging-system"),
    ("ISU Sports Rules", "https://www.isu.org/figure-skating-rules/"),
    ("U.S. Figure Skating Rules & Resources", "https://www.usfigureskating.org/about/rules"),
]

def simulate(skill, frames, difficulty, precision, fatigue):
    spec = MOVE_LIBRARY[skill]
    x = np.arange(frames)
    knee_target = 145 if spec["type"] == "jump" else 120
    trunk_target = 65 if spec["type"] == "spin" else 90
    knee = np.clip(knee_target + 14*np.sin(x/8) - difficulty*8 - fatigue*0.9 + np.random.normal(0, 1.6, frames), 35, 180)
    trunk = np.clip(trunk_target + 11*np.cos(x/11) + difficulty*5 - fatigue*0.4 + np.random.normal(0, 1.4, frames), 20, 150)
    quality = np.clip(100 - np.abs(knee-knee_target)*0.45 - np.abs(trunk-trunk_target)*0.35 + precision*1.5, 0, 100)
    df = pd.DataFrame({"frame": x+1, "knee": knee, "trunk": trunk, "quality": quality})
    execution = float(df["quality"].mean())
    goe = float(np.clip((execution - 70)/6, -5, 5))
    pcs = {
        "Skating Skills": round(np.clip(4 + execution/20, .25, 10), 2),
        "Composition": round(np.clip(4.2 + execution/21, .25, 10), 2),
        "Performance": round(np.clip(4.1 + execution/20.5, .25, 10), 2),
        "Interpretation": round(np.clip(4 + execution/22, .25, 10), 2),
    }
    total = round(spec["base"] + goe*0.1 + sum(pcs.values()), 2)
    return df, execution, goe, pcs, total

def plan_for(skill_type, phase):
    library = {
        "jump": {
            "On-Ice": ["Edge drills", "Takeoff entries", "3 clean jump reps", "Landing hold for 2 seconds"],
            "Off-Ice": ["Single-leg strength", "Snap-downs", "Core anti-rotation", "Plyometric control"],
            "Drills": ["Air-position squeeze", "Rotation timing", "Stick-the-landing drill", "Landing alignment"],
            "Competition Prep": ["Run-through timing", "Quality reps only", "Mental cue routine", "Recovery glide"],
            "Recovery": ["Light skating", "Mobility", "Breath reset", "Sleep prioritization"],
        },
        "spin": {
            "On-Ice": ["Entry control", "Centering", "Speed maintenance", "Exit balance"],
            "Off-Ice": ["Balance board", "Axis hold", "Hip-line drills", "Shoulder alignment"],
            "Drills": ["Spot checks", "Free-leg consistency", "Core compression", "Position switches"],
            "Competition Prep": ["Short spin sets", "Quality centering", "Relaxed breathing", "Confidence cue"],
            "Recovery": ["Mobility flow", "Hip release", "Ankle reset", "Easy edges"],
        },
        "steps": {
            "On-Ice": ["Edge sequence", "Turns", "Rhythm shift", "Speed changes"],
            "Off-Ice": ["Agility ladder", "Ankle stiffness", "Cadence work", "Footwork memory"],
            "Drills": ["Four-corner turns", "Count-based sequences", "Acceleration patterns", "Pattern holds"],
            "Competition Prep": ["Music timing", "Clean exits", "Step quality under fatigue", "Performance intent"],
            "Recovery": ["Easy stroking", "Range of motion", "Foot massage", "Breathing"],
        },
    }
    return library[skill_type][phase]

with st.sidebar:
    interface_language = st.selectbox("Language / اللغة", ["Arabic", "English"])
    t = TEXT[interface_language]
    st.divider()
    st.caption("World Champion Coach Pro v3")

st.markdown(f'<div class="hero"><h1>⛸️ {t["title"]}</h1><p>{t["subtitle"]}</p></div>', unsafe_allow_html=True)

tab_settings, tab_dashboard, tab_evaluation, tab_weekly, tab_judge, tab_rules = st.tabs([
    t["coach_settings"], t["dashboard"], t["evaluation"], t["weekly"], t["judge"], t["rules"]
])

with tab_settings:
    st.subheader(t["coach_settings"])
    st.caption(t["settings_note"])
    a, b = st.columns(2)
    with a:
        coach_name = st.text_input(t["coach_name"], value=st.session_state.get("coach_name", "Head Coach"))
        team_name = st.text_input(t["team_name"], value=st.session_state.get("team_name", "World Team"))
        discipline = st.selectbox(t["discipline"], ["Singles", "Pairs", "Ice Dance", "Synchronized Skating"])
    with b:
        athlete_name = st.text_input(t["athlete"], value=st.session_state.get("athlete_name", "Skater"))
        season_goal = st.text_input(t["season_goal"], value=st.session_state.get("season_goal", "World Championship Readiness"))
        next_competition = st.text_input(t["competition"], value=st.session_state.get("next_competition", "Next Competition"))
    if st.button(t["save"], type="primary"):
        st.session_state.update({
            "coach_name": coach_name, "team_name": team_name, "athlete_name": athlete_name,
            "season_goal": season_goal, "next_competition": next_competition,
        })
        st.success(t["saved"])

with tab_dashboard:
    st.subheader(t["dashboard"])
    coach = st.session_state.get("coach_name", "Head Coach")
    team = st.session_state.get("team_name", "World Team")
    athlete = st.session_state.get("athlete_name", "Skater")
    goal = st.session_state.get("season_goal", "World Championship Readiness")
    competition = st.session_state.get("next_competition", "Next Competition")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t["athlete"], athlete)
    c2.metric(t["coach_name"], coach)
    c3.metric(t["team_name"], team)
    c4.metric(t["season_goal"], goal)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(f"**{t['competition']}:** {competition}")
    st.write("Use the Evaluation tab to generate a coaching report, then review the weekly plan and judge simulation.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_evaluation:
    st.subheader(t["evaluation"])
    x1, x2, x3 = st.columns(3)
    with x1:
        skill = st.selectbox(t["skill"], list(MOVE_LIBRARY.keys()))
        phase = st.selectbox(t["phase"], ["On-Ice", "Off-Ice", "Drills", "Competition Prep", "Recovery"])
        weekly_focus = st.selectbox(t["weekly_focus"], ["Technical", "Performance", "Consistency", "Recovery"])
    with x2:
        frames = st.slider(t["frames"], 40, 360, 120, 10)
        difficulty = st.slider(t["difficulty"], 0.0, 2.0, 1.0, 0.1)
    with x3:
        precision = st.slider(t["precision"], 0.0, 10.0, 5.0, 0.5)
        fatigue = st.slider(t["fatigue"], 0.0, 10.0, 3.0, 0.5)

    if st.button(t["run"], type="primary", use_container_width=True):
        df, execution, goe, pcs, total = simulate(skill, frames, difficulty, precision, fatigue)
        spec = MOVE_LIBRARY[skill]
        plan = plan_for(spec["type"], phase)
        issues = []
        if execution < 60:
            issues.append("Major stability issue / مشكلة كبيرة في الثبات")
        if goe < 0:
            issues.append("Likely GOE deductions / خصومات GOE محتملة")
        if fatigue > 6:
            issues.append("High fatigue: reduce load and add recovery / إرهاق مرتفع: خفّض الحمل وأضف استشفاء")
        if precision < 5:
            issues.append("Precision below target / الدقة أقل من الهدف")

        st.session_state["analysis"] = {
            "df": df, "execution": execution, "goe": goe, "pcs": pcs, "total": total,
            "skill": skill, "phase": phase, "plan": plan, "issues": issues,
            "base": spec["base"], "focus": spec["focus"], "weekly_focus": weekly_focus,
        }

    analysis = st.session_state.get("analysis")
    if analysis:
        cols = st.columns(5)
        cols[0].metric(t["base"], analysis["base"])
        cols[1].metric(t["goe"], round(analysis["goe"], 2))
        cols[2].metric(t["execution"], f"{analysis['execution']:.1f}/100")
        cols[3].metric(t["pcs"], round(sum(analysis["pcs"].values()), 2))
        cols[4].metric(t["target"], analysis["total"])

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(analysis["df"]["frame"], analysis["df"]["knee"], label="Knee", linewidth=2)
        ax.plot(analysis["df"]["frame"], analysis["df"]["trunk"], label="Trunk", linewidth=2)
        ax.plot(analysis["df"]["frame"], analysis["df"]["quality"], label="Quality", linewidth=2)
        ax.set_xlabel("Frame")
        ax.grid(alpha=.25)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        st.markdown(f"### {t['plan']}")
        for item in analysis["plan"]:
            st.success(item)
        st.markdown(f"### {t['corrections']}")
        for item in analysis["issues"] or ["Performance is progressing well / الأداء يتقدم بشكل جيد"]:
            st.warning(item)

with tab_weekly:
    st.subheader(t["weekly"])
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Run an evaluation first / شغّل التقييم أولًا")
    else:
        plan = analysis["plan"]
        weekly = pd.DataFrame([
            ["Mon", "On-Ice", plan[0]],
            ["Tue", "Off-Ice", plan[1]],
            ["Wed", "Drills", plan[2]],
            ["Thu", "On-Ice", plan[0]],
            ["Fri", "Competition Prep", "Run-through + quality reps"],
            ["Sat", "Performance", "Full program simulation"],
            ["Sun", "Recovery", "Mobility + light skating"],
        ], columns=["Day", "Phase", "Main Work"])
        st.dataframe(weekly, use_container_width=True, hide_index=True)
        st.session_state["weekly"] = weekly
        st.download_button(t["download_weekly"], weekly.to_csv(index=False).encode("utf-8"), "weekly_plan.csv", "text/csv")

with tab_judge:
    st.subheader(t["judge"])
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Run an evaluation first / شغّل التقييم أولًا")
    else:
        judge = pd.DataFrame([
            ["Technical Value", analysis["base"]],
            ["GOE", round(analysis["goe"], 2)],
            ["Skating Skills", analysis["pcs"]["Skating Skills"]],
            ["Composition", analysis["pcs"]["Composition"]],
            ["Performance", analysis["pcs"]["Performance"]],
            ["Interpretation", analysis["pcs"]["Interpretation"]],
        ], columns=["Metric", "Score"])
        st.dataframe(judge, use_container_width=True, hide_index=True)
        st.caption("Training estimate only — not an official ISU score.")

with tab_rules:
    st.subheader(t["references"])
    st.write("The program uses a coaching approximation inspired by technical value, GOE and program-component concepts. Always confirm current rules in official documents.")
    for title, url in REFERENCES:
        st.markdown(f"- [{title}]({url})")

analysis = st.session_state.get("analysis")
if analysis:
    weekly = st.session_state.get("weekly")
    report = {
        "timestamp": datetime.now().isoformat(),
        "coach": st.session_state.get("coach_name", "Head Coach"),
        "team": st.session_state.get("team_name", "World Team"),
        "athlete": st.session_state.get("athlete_name", "Skater"),
        "skill": analysis["skill"],
        "phase": analysis["phase"],
        "base": analysis["base"],
        "goe": round(analysis["goe"], 2),
        "execution": round(analysis["execution"], 2),
        "pcs": analysis["pcs"],
        "total": analysis["total"],
        "focus": analysis["focus"],
        "training_plan": analysis["plan"],
        "issues": analysis["issues"],
        "weekly_plan": weekly.to_dict(orient="records") if weekly is not None else [],
        "references": REFERENCES,
    }
    st.divider()
    st.download_button(t["download_json"], json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"), "world_champion_coach_report.json", "application/json")
    st.download_button(t["download_csv"], analysis["df"].to_csv(index=False).encode("utf-8"), "analysis_data.csv", "text/csv")

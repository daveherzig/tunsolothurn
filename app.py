import base64
import json
import random
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from punch_analysis import (
    F_MAX, G, MASS_KG, MIN_GAP_S, PUNCH_THRESHOLD, SENSOR_MAX, HERO_LEVELS,
    accel_frame, add_magnitude, best_punch, detect_events, format_duration, hourly_record,
    load, merge_scores, person_best_scores, seconds, superhero_for,
)

st.set_page_config(page_title="Sensor Data Analytics", layout="wide")
logo_dir = Path(__file__).parent
PERSONS = ["Yuri Aebi", "Marlon Gebbia", "David Herzig", "Dominic Leuthardt", "Tim Leutwyler", "Markus Schlienger", "Nino Wyssmann", "Andreas Zink"]

def render_footer():
    st.divider()
    col_tun, col_fhnw, col_people = st.columns([1, 1, 1], vertical_alignment="center")
    col_tun.image(str(logo_dir / "tunSolothurn_logo_rgb_web_ok.svg"), width=250)
    with col_fhnw:
        _, mid, _ = st.columns([1, 2, 1])
        mid.image(str(logo_dir / "FHNW_Logo.svg.webp"))
    col_people.markdown(
        "<div style='text-align:right'><b>Authors</b><br>"
        + "<br>".join(PERSONS)
        + "</div>",
        unsafe_allow_html=True,
    )


def finish():
    render_footer()
    st.stop()


def merge_records(existing, incoming) -> list:
    merged = []
    seen = set()
    for records in (existing, incoming):
        if records is None:
            continue
        if not isinstance(records, list):
            records = [records]
        for record in records:
            signature = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if signature not in seen:
                merged.append(record)
                seen.add(signature)
    return merged


def render_record(record_slot, entries: list[dict]) -> None:
    if record_slot is None:
        return
    now = pd.Timestamp.now()
    if entries:
        newest = max(entries, key=lambda entry: entry["time"])
        previous = hourly_record(entries, now, exclude_name=newest["name"])
        record = hourly_record(entries, now)
    else:
        newest = previous = record = None

    if record is None:
        record_slot.info("In der letzten Stunde gibt es noch keinen Rekord.")
    elif previous is None or newest["score"] > previous["score"]:
        record_slot.success(
            f"Rekord der letzten Stunde: {record['score']:03d} — {record['name']}",
            icon="🏆",
        )
    else:
        record_slot.info(
            f"Rekord der letzten Stunde: {previous['score']:03d} — {previous['name']}",
            icon="🏆",
        )


view = st.session_state.get("view")
record_slot = None

if view is None:
    st.markdown(
        "<h3 style='text-align:center;margin:0'>Sensor Data Analytics</h3>",
        unsafe_allow_html=True,
    )
    st.divider()
    record_slot = st.empty()
    source = st.radio(
        "Data source",
        ["REST service", "Upload file"],
        horizontal=True,
        key="data_source",
    )
else:
    source = st.session_state.get("data_source", "REST service")


if "loaded_records" not in st.session_state:
    previous_records = merge_records(
        st.session_state.get("rest_records"),
        st.session_state.get("uploaded_records"),
    )
    if previous_records:
        st.session_state["loaded_records"] = previous_records

records = st.session_state.get("loaded_records")
try:
    if view is None and source == "Upload file":
        uploaded = st.file_uploader(
            "Upload a data.json file",
            type=["json"],
            key=f"uploader_{st.session_state.get('uploader_version', 0)}",
        )
        if uploaded is not None:
            st.session_state["loaded_records"] = merge_records(
                records, json.load(uploaded)
            )
        records = st.session_state.get("loaded_records")
    elif view is None:
        url = st.text_input("REST service URL", value="https://v002253.fhnw.ch/retrieve", placeholder="https://example.com/api/data")
        if st.button("Fetch data", disabled=not url):
            with st.spinner("Fetching data..."):
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                st.session_state["loaded_records"] = merge_records(
                    records, resp.json()
                )
            st.session_state.update(view=None, selected=None)
        records = st.session_state.get("loaded_records")
    else:
        records = st.session_state.get("loaded_records")
except requests.RequestException as e:
    st.error(f"REST call failed: {e}")
    finish()
except (json.JSONDecodeError, ValueError) as e:
    st.error(f"Response is not valid JSON: {e}")
    finish()

if records is None:
    render_record(record_slot, st.session_state.get("punch_history", []))
    st.info("Please upload a JSON file or fetch data from a REST service (data.json format).")
    finish()

try:
    df = load(records)
except (ValueError, TypeError) as e:
    st.error(f"Could not parse data: {e}")
    finish()


# ---------------------------------------------------------------------------
# Boxautomat: Schlagkraft aus Accelerometer-Daten
# ---------------------------------------------------------------------------


def remember_scores(entries: list[dict]) -> list[dict]:
    """Speichert die Bestwerte in der Session (UI-Zustand) – die Logik steckt in merge_scores."""
    st.session_state["punch_history"] = merge_scores(st.session_state.get("punch_history", []), entries)
    return st.session_state["punch_history"]


def display_html(value: int) -> str:
    return (
        "<div style='display:flex;justify-content:center;margin:1rem 0'>"
        "<div style='background:#111;border:6px solid #444;border-radius:16px;"
        "padding:1rem 3rem;box-shadow:inset 0 0 25px #000'>"
        f"<span style='font-family:\"Courier New\",monospace;font-size:7rem;"
        "font-weight:bold;color:#ff2a2a;letter-spacing:0.5rem;"
        f"text-shadow:0 0 10px #ff2a2a,0 0 30px #ff0000'>{value:03d}</span>"
        "</div></div>"
    )


def gender_for_person(person_df: pd.DataFrame) -> str:
    if "gender" in person_df:
        genders = person_df["gender"].dropna()
        if not genders.empty:
            value = str(genders.iloc[0]).strip().lower()
            if value in ("male", "m", "männlich", "maennlich", "man"):
                return "Männlich"
            if value in ("female", "f", "w", "weiblich", "woman"):
                return "Weiblich"
            return "Divers (gemischt)"
    return "Männlich"


# Superhelden-Bewertung nach Score (untere Grenze, Held)

HERO_ICONS = {
    "Spiderman": ("spiderman-icon-new.png", "image/png"),
    "Batman": ("batman-icon-new.png", "image/png"),
    "Wolverine": ("wolverine-icon-new.png", "image/png"),
    "Deadpool": ("deadpool-icon-new.png", "image/png"),
    "Ironman": ("ironman-icon-new.png", "image/png"),
    "Superman": ("superman-icon-new.png", "image/png"),
    "Wonder Woman": ("wonderwoman-icon.png", "image/png"),
    "She-Hulk": ("shehulk-icon.png", "image/png"),
    "Captain Marvel": ("captainmarvel-icon.png", "image/png"),
    "Catwoman": ("catwoman-icon.png", "image/png"),
    "Gamora": ("gomora-icon.png", "image/png"),
    "Black Widow": ("blackwidow-icon.png", "image/png"),
}


def hero_icon_html(hero: str, size: int = 260) -> str:
    filename, mime = HERO_ICONS[hero]
    path = logo_dir / filename
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode()
    return (
        f"<img src='data:{mime};base64,{encoded}' alt='{hero}' "
        f"style='width:{size}px;height:{size}px;object-fit:contain'>"
    )


if "name" not in df:
    df["name"] = ""
df["name"] = df["name"].fillna("")
view = st.session_state.get("view")
selected = st.session_state.get("selected")
if view is not None and selected not in set(df["name"]):
    st.session_state.update(view=None, selected=None)
    st.rerun()


entries = remember_scores(person_best_scores(df))
render_record(record_slot, entries)


if view is None or selected not in set(df["name"]):
    if st.button("Clear data", help="Remove all loaded data from the table"):
        st.session_state.pop("loaded_records", None)
        st.session_state.pop("rest_records", None)
        st.session_state.pop("uploaded_records", None)
        st.session_state["uploader_version"] = st.session_state.get("uploader_version", 0) + 1
        st.session_state.update(view=None, selected=None)
        st.rerun()
    header = st.columns([2, 4, 1, 2, 2])
    for col, title in zip(header, ["Name", "Time", "Score", "Analytics", "Boxautomat"]):
        col.markdown(f"**{title}**")
    for person, person_df in df.groupby("name", sort=True):
        best, punches = best_punch(person_df)
        score = int(best["score"]) if best is not None else None
        row = st.columns([2, 4, 1, 2, 2], vertical_alignment="center")
        row[0].write(person or "(no name)")
        row[1].write(format_duration(person_df))
        row[2].write(f"{score:03d}" if score is not None else "–")
        if row[3].button("📊 Analytics", key=f"analytics_{person}"):
            st.session_state.update(view="analytics", selected=person)
            st.rerun()
        if row[4].button("🥊 Boxautomat", key=f"punch_{person}"):
            gender = gender_for_person(person_df)
            punch_gender = gender
            st.session_state.update(
                view="punch", selected=person, punch_gender=punch_gender
            )
            st.rerun()
    finish()

with st.container(horizontal=True, gap="xxsmall"):
    if st.button("Back to overview", width="content"):
        st.session_state.update(view=None, selected=None)
        st.rerun()

    if view == "punch":
        if st.button("📊 Analytics", width="content"):
            st.session_state["view"] = "analytics"
            st.rerun()
    elif st.button("🥊 Boxautomat", width="content"):
        person_df = df[df["name"] == selected]
        gender = gender_for_person(person_df)
        punch_gender = gender
        st.session_state.update(view="punch", punch_gender=punch_gender)
        st.rerun()

if view == "punch":
    person_df = df[df["name"] == selected]
    acc = accel_frame(person_df)
    best, punches = best_punch(person_df)
    if best is None:
        st.warning("Kein Schlag erkannt (keine Accelerometer-Daten oder alle Werte unter dem Schwellwert).")
        finish()

    score = int(best["score"])

    score_col, hero_col = st.columns(2, vertical_alignment="center")
    score_col.markdown(
        f"<h2 style='text-align:center'>🥊 Boxautomat: {selected or '(no name)'}</h2>",
        unsafe_allow_html=True,
    )
    # Hochzählen wie an einem echten Automaten
    screen = score_col.empty()
    steps = 30
    for i in range(1, steps + 1):
        screen.markdown(display_html(int(score * i / steps)), unsafe_allow_html=True)
        time.sleep(0.03)
    score_col.progress(score / 999)
    gender = st.session_state.get("punch_gender", "Männlich")
    hero = superhero_for(score, gender)
    hero_col.markdown(
        f"<h1 style='text-align:center;font-weight:bold'>{hero}</h1>"
        f"<div style='display:flex;justify-content:center'>{hero_icon_html(hero)}</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    levels = HERO_LEVELS[gender]
    score_ranges = [
        f"{hero_name} {limit}–{levels[index - 1][0] - 1 if index else 999}"
        for index, (limit, hero_name) in enumerate(levels)
    ]
    st.caption(" · ".join(score_ranges))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kraft (geschätzt)", f"{best['force_N']:.0f} N")
    c2.metric("Spitzenbeschleunigung", f"{best['a_dyn']:.1f} m/s²")
    c3.metric("Erkannte Schläge", len(punches))
    c4.metric("Bester Schlag", f"#{int(best['punch'])}")
    if best["a_dyn"] + G >= 0.97 * SENSOR_MAX:
        st.warning("Der Sensor war am Anschlag (±16 g). Der echte Wert war vermutlich noch höher.")

    st.markdown("**Schlagverlauf**")
    # Zeitachse in Sekunden seit Aufnahmebeginn (lesbarer als Uhrzeiten mit Millisekunden)
    t0 = acc["friendlyTimeStamp"].min()

    def with_seconds(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(t_s=(frame["friendlyTimeStamp"] - t0).dt.total_seconds())

    x_axis = alt.X("t_s:Q", title="Zeit seit Aufnahmebeginn (s)")
    # Bei wenigen Messwerten (z. B. alle 200 ms) die einzelnen Punkte sichtbar machen
    sparse = bool(np.median(np.diff(seconds(acc))) > 0.05) if len(acc) > 1 else True
    line = alt.Chart(with_seconds(acc)).mark_line(point=sparse).encode(
        x=x_axis,
        y=alt.Y("a_dyn:Q", title="Beschleunigung ohne g (m/s²)"),
    )
    rule = alt.Chart(pd.DataFrame({"y": [PUNCH_THRESHOLD]})).mark_rule(
        strokeDash=[4, 4], color="gray"
    ).encode(y="y:Q")
    points = alt.Chart(with_seconds(punches)).mark_point(size=120, filled=True, color="red").encode(
        x="t_s:Q",
        y="a_dyn:Q",
        tooltip=["punch", "score", alt.Tooltip("force_N:Q", format=".0f"),
                 alt.Tooltip("t_s:Q", title="Zeit (s)", format=".2f")],
    )
    chart = line + rule + points
    events = detect_events(acc)
    returns = events[events["kind"] == "Rückbewegung"] if not events.empty else events
    if not returns.empty:
        chart += alt.Chart(with_seconds(returns)).mark_point(
            size=120, shape="cross", color="gray"
        ).encode(
            x="t_s:Q",
            y="a_dyn:Q",
            tooltip=["kind", alt.Tooltip("a_dyn:Q", format=".1f"),
                     alt.Tooltip("t_s:Q", title="Zeit (s)", format=".2f")],
        )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"🔴 Schlag · ✚ Rückbewegung (ignoriert): {len(returns)} erkannt"
    )

    st.markdown("**Alle Schläge**")
    st.dataframe(
        punches.rename(columns={
            "punch": "Schlag", "friendlyTimeStamp": "Zeit",
            "a_dyn": "a (m/s²)", "force_N": "Kraft (N)", "score": "Score",
        }).round({"a (m/s²)": 1, "Kraft (N)": 0}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"Verwendeter Sensor: {acc['sensorId'].iloc[0]}. "
        f"Kraft = {MASS_KG} kg × Spitzenbeschleunigung (ohne Schwerkraft). "
        f"Score = Kraft / {F_MAX:.0f} N × 999. "
        f"Schläge werden ab {PUNCH_THRESHOLD:.0f} m/s² und mit mindestens {MIN_GAP_S} s Abstand erkannt."
    )
    finish()

st.markdown(
    f"<h2 style='text-align:center;margin-bottom:2rem'>📊 Analytics: {selected or '(no name)'}</h2>",
    unsafe_allow_html=True,
)
person_df = df[df["name"] == selected]
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(person_df))
if "deviceId" in person_df:
    c2.metric("Devices", person_df["deviceId"].nunique())
elif "device" in person_df:
    c2.metric("Devices", person_df["device"].nunique())
if "sensorId" not in person_df:
    st.warning("The data contains no sensorId field.")
    finish()
all_sensors = sorted(person_df["sensorId"].unique())
c3.metric("Sensors", len(all_sensors))

sensors = st.multiselect("Sensors to analyze", all_sensors, default=all_sensors[:1])
if not sensors:
    st.info("Select at least one sensor.")
    finish()
person_df = add_magnitude(person_df[person_df["sensorId"].isin(sensors)])
has_time = "friendlyTimeStamp" in person_df
channel_cols = [c for c in ("x", "y", "z") + tuple(f"data_{i}" for i in range(3, 10)) if c in person_df]


def sensor_frame(sensor: str) -> pd.DataFrame:
    sub = person_df[person_df["sensorId"] == sensor]
    return sub.dropna(axis=1, how="all")


tab_time, tab_compare, tab_dist, tab_stats, tab_raw = st.tabs(
    ["Time series", "Compare sensors", "Distribution", "Statistics", "Raw data"]
)

with tab_time:
    if not has_time:
        st.info("No timestamps available.")
    for sensor in sensors if has_time else []:
        sub = sensor_frame(sensor)
        cols = [c for c in channel_cols if c in sub]
        st.markdown(f"**{sensor}**")
        picked = st.multiselect("Channels", cols, default=cols, key=f"ch_{sensor}")
        if picked:
            st.line_chart(sub.set_index("friendlyTimeStamp")[picked])

with tab_compare:
    if not has_time or "magnitude" not in person_df:
        st.info("Comparison needs timestamps and 3-axis data.")
    else:
        st.caption("Magnitude of the first three channels of each selected sensor.")
        pivot = person_df.pivot_table(
            index="friendlyTimeStamp", columns="sensorId", values="magnitude", aggfunc="mean"
        )
        st.line_chart(pivot)
        st.markdown("**Maximum magnitude per sensor**")
        st.bar_chart(person_df.groupby("sensorId")["magnitude"].max())

with tab_dist:
    sensor = st.selectbox("Sensor", sensors, key="dist_sensor")
    sub = sensor_frame(sensor)
    cols = [c for c in channel_cols + ["magnitude"] if c in sub]
    channel = st.selectbox("Channel", cols, key="dist_channel")
    bins = st.slider("Bins", 5, 100, 30)
    counts = pd.cut(sub[channel], bins=bins).value_counts(sort=False)
    counts.index = [f"{iv.mid:.3f}" for iv in counts.index]
    st.bar_chart(counts)

with tab_stats:
    for sensor in sensors:
        sub = sensor_frame(sensor)
        cols = [c for c in channel_cols + ["magnitude"] if c in sub]
        st.markdown(f"**{sensor}**")
        st.dataframe(sub[cols].describe(), use_container_width=True)

with tab_raw:
    st.dataframe(person_df, use_container_width=True)


render_footer()

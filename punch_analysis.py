"""Berechnungslogik für den virtuellen Boxautomaten.

Dieses Modul enthält KEIN Streamlit (kein `st.`). Es nimmt Daten entgegen,
rechnet und gibt Ergebnisse zurück. Die Darstellung passiert in app.py.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Daten laden und aufbereiten
# ---------------------------------------------------------------------------
def load(records) -> pd.DataFrame:
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)
    if "data" in df:
        axes = pd.DataFrame(df["data"].tolist(), index=df.index)
        axes.columns = [f"data_{c}" for c in axes.columns]
        axes = axes.rename(columns={"data_0": "x", "data_1": "y", "data_2": "z"})
        df = pd.concat([df.drop(columns="data"), axes], axis=1)
    if "friendlyTimeStamp" in df:
        df["friendlyTimeStamp"] = pd.to_datetime(df["friendlyTimeStamp"], errors="coerce")
        df = df.sort_values("friendlyTimeStamp").reset_index(drop=True)
    return df


def format_duration(frame: pd.DataFrame) -> str:
    if "friendlyTimeStamp" not in frame or frame["friendlyTimeStamp"].isna().all():
        return "-"
    start, end = frame["friendlyTimeStamp"].min(), frame["friendlyTimeStamp"].max()
    return f"{start:%H:%M:%S} – {end:%H:%M:%S} ({(end - start).total_seconds():.1f} s)"


def add_magnitude(frame: pd.DataFrame) -> pd.DataFrame:
    axes = [a for a in ("x", "y", "z") if a in frame]
    if len(axes) == 3:
        frame = frame.assign(magnitude=(frame[axes] ** 2).sum(axis=1) ** 0.5)
    return frame


# ---------------------------------------------------------------------------
# Boxautomat: Parameter
# ---------------------------------------------------------------------------
G = 9.81                    # Erdbeschleunigung in m/s²
MASS_KG = 2.5               # effektive Masse von Faust + Arm (fixe Annahme)
SENSOR_MAX = 16 * G         # Messbereich des Sensors (±16 g) ≈ 157 m/s²
F_MAX = MASS_KG * SENSOR_MAX  # grösste messbare Kraft -> entspricht 999 Punkten
PUNCH_THRESHOLD = 20.0      # ab dieser Beschleunigung (ohne g) zählt es als Bewegung
MIN_GAP_S = 0.4             # Werte mit weniger Abstand gehören zur selben Bewegung
RETURN_WINDOW_S = 1.0       # schwächere Bewegung so kurz nach einem Schlag = Rückbewegung


# ---------------------------------------------------------------------------
# Boxautomat: Schläge erkennen und bewerten
# ---------------------------------------------------------------------------
def accel_frame(person_df: pd.DataFrame) -> pd.DataFrame:
    """Beschleunigungsdaten, zeitlich sortiert, mit dynamischer Beschleunigung a_dyn (ohne g).

    1. Wahl: Accelerometer (inkl. Schwerkraft) -> g wird abgezogen
    2. Wahl: linear_acceleration (bereits ohne Schwerkraft) -> Betrag direkt verwenden
    """
    if "sensorId" not in person_df or not {"x", "y", "z"} <= set(person_df):
        return pd.DataFrame()
    ids = person_df["sensorId"].astype(str).str.lower()
    is_accel = ids.str.contains("accelerometer") & ~ids.str.contains("uncalibrated")
    is_linear = ids.str.contains("linear")

    if (is_accel & ~is_linear).any():
        acc, subtract_g = person_df[is_accel & ~is_linear], True
    elif is_linear.any():
        acc, subtract_g = person_df[is_linear], False
    else:
        return pd.DataFrame()

    sort_col = "timestamp" if "timestamp" in acc else "friendlyTimeStamp"
    acc = acc.sort_values(sort_col).reset_index(drop=True)
    magnitude = np.sqrt((acc[["x", "y", "z"]] ** 2).sum(axis=1))
    a_dyn = (magnitude - G).clip(lower=0) if subtract_g else magnitude
    return acc.assign(a_dyn=a_dyn)


def seconds(acc: pd.DataFrame) -> np.ndarray:
    """Zeit in Sekunden (nur Differenzen sind relevant, nicht der absolute Wert)."""
    if "timestamp" in acc and acc["timestamp"].notna().all():
        return (acc["timestamp"] / 1e9).to_numpy()           # Sensor-Zeit in Nanosekunden
    # Fallback (z. B. Daten über backend.py ohne "timestamp"):
    # Abstand zum ersten Zeitpunkt -> funktioniert unabhängig von der internen Auflösung
    ts = acc["friendlyTimeStamp"]
    return (ts - ts.min()).dt.total_seconds().to_numpy()


def detect_events(acc: pd.DataFrame) -> pd.DataFrame:
    """Teilt die Aufnahme in Bewegungen auf und unterscheidet Schlag und Rückbewegung.

    1. Alle Werte über dem Schwellwert, die weniger als MIN_GAP_S auseinanderliegen,
       bilden eine Bewegung.
    2. Eine Bewegung ist eine Rückbewegung, wenn sie
         - höchstens RETURN_WINDOW_S nach dem Ende eines Schlags beginnt UND
         - schwächer ist als dieser Schlag.
       Alle anderen Bewegungen sind Schläge.
    """
    if acc.empty:
        return pd.DataFrame()
    t = seconds(acc)
    above = np.where(acc["a_dyn"].to_numpy() >= PUNCH_THRESHOLD)[0]
    if len(above) == 0:
        return pd.DataFrame()

    # 1. Bewegungen bilden
    groups, current = [], [above[0]]
    for i in above[1:]:
        if t[i] - t[current[-1]] > MIN_GAP_S:
            groups.append(current)
            current = []
        current.append(i)
    groups.append(current)

    a = acc["a_dyn"].to_numpy()
    rows = []
    last_punch = None                                   # zuletzt erkannter echter Schlag
    for idx in groups:
        peak = idx[int(np.argmax(a[idx]))]
        event = {"peak": peak, "a_dyn": a[peak], "start": t[idx[0]], "end": t[idx[-1]]}

        # 2. Kurz nach einem Schlag und schwächer -> Rückbewegung
        is_return = (
            last_punch is not None
            and event["start"] - last_punch["end"] <= RETURN_WINDOW_S
            and event["a_dyn"] < last_punch["a_dyn"]
        )
        event["kind"] = "Rückbewegung" if is_return else "Schlag"
        if not is_return:
            last_punch = event
        rows.append(event)

    events = pd.DataFrame(rows)
    events.insert(0, "friendlyTimeStamp", acc.loc[events["peak"], "friendlyTimeStamp"].values)
    events["force_N"] = MASS_KG * events["a_dyn"]
    events["score"] = events["force_N"].map(force_to_score)
    return events.drop(columns=["peak", "start", "end"])


def detect_punches(acc: pd.DataFrame) -> pd.DataFrame:
    """Nur die echten Schläge (ohne Rückbewegungen), durchnummeriert."""
    events = detect_events(acc)
    if events.empty:
        return pd.DataFrame()
    punches = events[events["kind"] == "Schlag"].drop(columns="kind").copy()
    punches.insert(0, "punch", range(1, len(punches) + 1))
    return punches.reset_index(drop=True)


def force_to_score(force_n: float) -> int:
    return int(round(min(force_n / F_MAX, 1.0) * 999))


def best_punch(person_df: pd.DataFrame):
    punches = detect_punches(accel_frame(person_df))
    if punches.empty:
        return None, punches
    return punches.loc[punches["score"].idxmax()], punches


# ---------------------------------------------------------------------------
# Bestenliste / Rekord der letzten Stunde
# ---------------------------------------------------------------------------
def person_best_scores(frame: pd.DataFrame) -> list[dict]:
    """Bester Schlag pro Person: Name, Score und Zeitpunkt."""
    entries = []
    for person, person_df in frame.groupby("name", sort=True):
        best, _ = best_punch(person_df)
        if best is None or pd.isna(best["friendlyTimeStamp"]):
            continue
        entries.append({
            "name": person or "(no name)",
            "score": int(best["score"]),
            "time": best["friendlyTimeStamp"],
        })
    return entries


def merge_scores(history: list[dict], entries: list[dict]) -> list[dict]:
    """Behält pro Person den höheren Score (reine Logik, ohne session_state)."""
    merged = {entry["name"]: entry for entry in history}
    for entry in entries:
        old = merged.get(entry["name"])
        if old is None or entry["score"] >= old["score"]:
            merged[entry["name"]] = entry
    return list(merged.values())


def hourly_record(entries: list[dict], now: pd.Timestamp, exclude_name: str | None = None):
    cutoff = now - pd.Timedelta(hours=1)
    recent = [
        entry for entry in entries
        if entry["time"] >= cutoff and entry["name"] != exclude_name
    ]
    if not recent:
        return None
    return max(recent, key=lambda entry: entry["score"])


# ---------------------------------------------------------------------------
# Superhelden-Bewertung
# ---------------------------------------------------------------------------
HERO_LEVELS = {
    "Männlich": [
        (750, "Superman"),
        (600, "Ironman"),
        (500, "Deadpool"),
        (350, "Wolverine"),
        (200, "Batman"),
        (0, "Spiderman"),
    ],
    "Weiblich": [
        (750, "Wonder Woman"),
        (600, "She-Hulk"),
        (500, "Captain Marvel"),
        (350, "Catwoman"),
        (200, "Gamora"),
        (0, "Black Widow"),
    ],
}


def superhero_for(score: int, gender: str) -> str:
    return next(hero for limit, hero in HERO_LEVELS[gender] if score >= limit)
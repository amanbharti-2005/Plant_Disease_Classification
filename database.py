"""
Persistent memory for AgroVision — every diseased-plant scan gets stored
here permanently (not just in Streamlit's session state, which resets
on refresh). This is what powers:
  - the affected-areas map
  - the report history
  - the rover's "go back and check this spot" API

Uses SQLite: a single file (agrovision.db), no separate database server
to install or run.
"""
import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager
import config

DB_PATH = config.DB_PATH


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                species TEXT,
                condition TEXT,
                severity TEXT,
                confidence REAL,
                treatment TEXT,
                recommendations TEXT,
                thumbnail TEXT,
                weed_count INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active'
            )
        """)


def save_scan(species, condition, severity, confidence, treatment,
              recommendations, thumbnail=None, latitude=None, longitude=None,
              weed_count=0):
    """
    Called after every successful diagnosis. Only meant to be called for
    real detections (not 'No Target Object Spotted' / 'Invalid Image' —
    those aren't worth remembering).

    status:
      'healthy'  -> plant was fine, nothing to revisit
      'active'   -> disease found, not yet treated — shows up on the map
                    and in /affected-areas for the rover to revisit

    weed_count is stored regardless of disease status — a healthy crop
    surrounded by weeds is still something worth flagging for the rover.
    """
    status = "healthy" if condition.strip().lower() == "healthy" else "active"
    with _connect() as conn:
        cursor = conn.execute("""
            INSERT INTO scans
                (timestamp, latitude, longitude, species, condition, severity,
                 confidence, treatment, recommendations, thumbnail, weed_count, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(), latitude, longitude, species, condition,
            severity, confidence, treatment, json.dumps(recommendations),
            thumbnail, weed_count, status
        ))
        return cursor.lastrowid


def get_affected_areas():
    """
    For the rover's navigation code to poll: every location that still
    has an unresolved disease flagged. Only returns entries that HAVE
    coordinates — a scan with no GPS fix can't be revisited.
    """
    with _connect() as conn:
        rows = conn.execute("""
            SELECT id, timestamp, latitude, longitude, species, condition,
                   severity, confidence
            FROM scans
            WHERE status = 'active' AND latitude IS NOT NULL AND longitude IS NOT NULL
            ORDER BY timestamp DESC
        """).fetchall()
        return [dict(r) for r in rows]


def get_all_reports(species=None, severity=None, status=None, limit=200):
    query = "SELECT * FROM scans WHERE 1=1"
    params = []
    if species:
        query += " AND species = ?"
        params.append(species)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["recommendations"] = json.loads(d["recommendations"]) if d["recommendations"] else []
            results.append(d)
        return results


def get_report(report_id):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["recommendations"] = json.loads(d["recommendations"]) if d["recommendations"] else []
        return d


def mark_resolved(report_id):
    """Called when the rover (or a human) confirms the affected plant
    has been treated — removes it from /affected-areas going forward."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE scans SET status = 'resolved' WHERE id = ?", (report_id,)
        )
        return cursor.rowcount > 0


def get_weed_summary():
    """Answers 'which crop had how many weeds' — sums weed_count grouped
    by species across all scans, plus how many separate scan locations
    reported at least one weed."""
    with _connect() as conn:
        rows = conn.execute("""
            SELECT species,
                   SUM(weed_count) AS total_weeds,
                   COUNT(CASE WHEN weed_count > 0 THEN 1 END) AS spots_with_weeds,
                   COUNT(*) AS total_scans
            FROM scans
            WHERE species IS NOT NULL
            GROUP BY species
            ORDER BY total_weeds DESC
        """).fetchall()
        return [dict(r) for r in rows]


init_db()
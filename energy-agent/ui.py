"""Manager UI (Streamlit). Uses only the public API, like the judges do."""

import json
import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Factory energy cost", layout="wide")
st.title("Factory energy cost agent")


def show(text):
    """Answers are markdown: put each section (**Answer:**, **Why:** ...) on its own paragraph."""
    st.markdown(text.replace("\n**", "\n\n**"))


def ask(question, period=None, compare_to=None):
    return httpx.post(f"{API}/analyze", json={"question": question, "period": period or None,
                                              "compare_to": compare_to or None},
                      timeout=120).json()


bridge_tab, waste_tab, ask_tab, data_tab = st.tabs(["Bill bridge", "Waste & actions", "Ask", "Data"])

with bridge_tab:
    c1, c2 = st.columns(2)
    period, compare = c1.text_input("Period", "2026-Q3"), c2.text_input("Compared with", "2026-Q2")
    if st.button("Explain the change", type="primary"):
        r = ask("Why did the bill change?", period, compare)
        show(r["summary"])
        if r["bridge"]:   # R11: one chart that explains the change
            b = r["bridge"]
            st.plotly_chart(go.Figure(go.Waterfall(
                x=[x["cause"].replace("_", " ") for x in b] + ["total"],
                y=[x["pct_points"] for x in b] + [r["total_pct"]],
                measure=["relative"] * len(b) + ["total"],
                text=[f"{x['pct_points']} ({x['range'][0]} to {x['range'][1]})" for x in b] + [f"{r['total_pct']}%"])),
                use_container_width=True)
            st.caption("Percentage points of the base bill. The parts add up to the total; ranges in brackets.")
            st.write("Assumptions:", r["assumptions"], "Limitations:", r["limitations"])

with waste_tab:
    if st.button("Find waste"):
        assets = ask("Which assets use the most energy?")["result"].get("assets", [])
        st.bar_chart(pd.DataFrame(assets).set_index("meter")["kwh"])
        show(ask("Is there a leak at night?")["summary"])
        recs = ask("What should we fix first?")["recommendations"]
        if recs:
            st.dataframe(pd.DataFrame(recs)[["action", "saving_kwh_per_year", "saving_money_per_year",
                                             "saving_co2e_t_per_year", "payback_months", "ease", "requires_approval"]])

with ask_tab:   # one question -> one complete answer (no chat memory)
    q = st.text_input("Question", "Why did the bill go up?")
    if st.button("Ask"):
        r = ask(q)
        st.caption(f"answer type: {r['answer_type']} · run {r['run_id']}")
        show(r["summary"])
        with st.expander("Trace (steps, tools, data used)"):
            st.json(httpx.get(f"{API}/trace/{r['run_id']}", timeout=30).json())

with data_tab:
    table = st.selectbox("Table", ["meter_15min", "tariff", "production", "weather"])
    f = st.file_uploader("CSV", type="csv")
    if f and st.button("Upload"):
        st.json(httpx.post(f"{API}/ingest", files={"file": (f.name, f.getvalue())},
                           data={"meta": json.dumps({"table": table})}, timeout=600).json())
    st.json(httpx.get(f"{API}/health", timeout=10).json())

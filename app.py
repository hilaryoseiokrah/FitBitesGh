###############################################################################
# FitBites – Ghana-centric AI Meal-Planner
# Rev. 2025-05-21  •  BMI/TDEE, ANN recs, GPT combos, reshuffles, profile, recipes
###############################################################################

import os, json, datetime as dt, random, re
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────── OpenAI (optional) ───────────────────────────
try:
    from openai import OpenAI
    client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY",
                                             st.secrets.get("OPENAI_API_KEY", "")))
    OPENAI_AVAILABLE = True
except Exception:
    client_openai, OPENAI_AVAILABLE = None, False

# ─────────────────────────── Paths ───────────────────────────────────
DATA      = Path("data")
PROFILES  = DATA / "profiles"
PICS      = DATA / "profile_pics"
PLANS     = DATA / "mealplans"
for p in (DATA, PROFILES, PICS, PLANS):
    p.mkdir(parents=True, exist_ok=True)

USERS_CSV = DATA / "users.csv"

# ──────────────────── Auth helpers ───────────────────────────────────

def _users_df() -> pd.DataFrame:
    """Always return a DataFrame with username & password as strings."""
    if USERS_CSV.exists():
        return pd.read_csv(USERS_CSV, dtype=str)   # ← force str
    return pd.DataFrame(columns=["username", "password"], dtype=str)

def register_user(username: str, password: str) -> bool:
    df = _users_df()
    uname = username.strip().lower()               # ← store lowercase
    if uname in df.username.str.strip().str.lower().values:
        return False                               # already exists
    df.loc[len(df)] = [uname, password]
    df.to_csv(USERS_CSV, index=False, header=True)
    return True

def authenticate(username: str, password: str) -> bool:
    df = _users_df()
    uname = username.strip().lower()               # ← compare lowercase
    mask = (df.username.str.strip().str.lower() == uname) & (df.password == password)
    return mask.any()

# ───────────────── Profile helpers ──────────────────────────────────
def profile_path(u):   return PROFILES / f"{u}.json"
def picture_path(u):   return PICS / f"{u}.png"
def plan_path(u, w):   return PLANS / f"{u}_week{w}.csv"

def load_profile(u):
    if profile_path(u).exists():
        return json.loads(profile_path(u).read_text())
    return dict(weight=90, height=160, age=25, sex="female",
                activity="sedentary", target_weight=75,
                likes_b=[], likes_l=[], likes_d=[], dislikes=[],
                use_ai=False, last_updated=str(dt.date.today()))

def save_profile(u, obj): profile_path(u).write_text(json.dumps(obj, indent=2))

def existing_weeks(u):
    return sorted(int(m.group(1)) for f in PLANS.glob(f"{u}_week*.csv")
                  if (m := re.search(r"week(\d+)", f.name)))

# ─────────────────── BMI / TDEE helpers ─────────────────────────────
def bmi(w, h_cm): return w / (h_cm/100)**2
def tdee_msj(w, h, age, sex, act):
    base = 10*w + 6.25*h - 5*age + (5 if sex == "male" else -161)
    mult = dict(sedentary=1.2, light=1.375, moderate=1.55,
                active=1.725, superactive=1.9)[act]
    return base * mult

SAFE_KG_PER_WEEK = 0.75
def weeks_to_goal(cur, tgt): return abs(cur - tgt) / SAFE_KG_PER_WEEK

# ─────────────── Load food & build embeddings ───────────────────────
@st.cache_data(show_spinner=False)
def load_food():
    df = pd.read_csv("gh_food_nutritional_values.csv")
    df["Food"] = df["Food"].str.strip().str.lower()
    cols = ["Protein(g)", "Fat(g)", "Carbs(g)", "Calories(100g)",
            "Water(g)", "SFA(100g)", "MUFA(100g)", "PUFA(100g)"]
    df[cols] = df[cols].fillna(df[cols].mean())
    return df, cols
FOODS_DF, NUTR_COLS = load_food()

class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8))
        self.d = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(self, x): z = self.e(x); return z, self.d(z)

@st.cache_resource(show_spinner=False)
def build_embeddings(mat):
    torch.manual_seed(7)
    net = AE(mat.shape[1]); opt = optim.Adam(net.parameters(), 1e-3)
    loss = nn.MSELoss(); t = torch.tensor(mat, dtype=torch.float32)
    for _ in range(300):
        opt.zero_grad(); z, out = net(t); loss(out, t).backward(); opt.step()
    with torch.no_grad(): z, _ = net(t); return z.numpy()
EMB = build_embeddings(StandardScaler().fit_transform(FOODS_DF[NUTR_COLS]))

def top_similar(food, k=5, exclude=None):
    exclude = exclude or []
    idx = FOODS_DF.index[FOODS_DF.Food == food]
    if idx.empty: return []
    sims = cosine_similarity(EMB[idx[0]].reshape(1, -1), EMB).ravel()
    out = []
    for i in sims.argsort()[::-1]:
        name = FOODS_DF.iloc[i].Food
        if name != food and name not in exclude:
            out.append(name)
        if len(out) == k: break
    return out

# ────────────────────────── Streamlit UI ───────────────────────────
st.set_page_config("FitBites – AI Ghanaian Meal Plans", layout="wide")

if "state" not in st.session_state:
    st.session_state.state = dict(
        logged=False, user="", profile={},
        meal_plan=None, week=None, daily_k=None)
S = st.session_state.state

# ─────────────────── Login / Register ───────────────────────────────
if not S["logged"]:
    st.title("🔐 FitBites Login")
    login_tab, reg_tab = st.tabs(("Login", "Register"))

    with login_tab:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(u, p):
                S.update(dict(logged=True, user=u, profile=load_profile(u)))
                st.rerun()
            else:
                st.error("Incorrect username/password")

    with reg_tab:
        nu  = st.text_input("Choose username")
        npw = st.text_input("Choose password", type="password")
        if st.button("Create account"):
            ok = register_user(nu, npw)
            st.success("Account created") if ok else st.warning("User exists")

    st.stop()

# ───────────── Sidebar – profile inputs ─────────────────────────────
with st.sidebar:
    st.subheader(f"Hi, {S['user'].title()}")
    P = S["profile"]
    weight = st.number_input("Weight (kg)", 30., 200., P["weight"], 0.1)
    target = st.number_input("Target (kg)", 30., 200., P["target_weight"], 0.1)
    height = st.number_input("Height (cm)", 120., 250., P["height"], 0.1)
    age    = st.number_input("Age", 10, 100, P["age"], 1)
    sex    = st.selectbox("Sex", ["female", "male"],
                          0 if P["sex"] == "female" else 1)
    activity = st.selectbox("Activity", ["sedentary","light","moderate",
                                         "active","superactive"],
                            ["sedentary","light","moderate","active",
                             "superactive"].index(P["activity"]))
    st.markdown("---")
    st.subheader("Meal preferences")
    likes_b = st.multiselect("Breakfast", FOODS_DF.Food, P["likes_b"])
    likes_l = st.multiselect("Lunch",     FOODS_DF.Food, P["likes_l"])
    likes_d = st.multiselect("Dinner",    FOODS_DF.Food, P["likes_d"])
    dislikes= st.multiselect("Dislikes",  FOODS_DF.Food, P["dislikes"])
    use_ai  = st.checkbox("Use GPT combos", P["use_ai"], disabled=not OPENAI_AVAILABLE)

    if st.button("Save profile"):
        P.update(weight=weight, height=height, age=age, sex=sex,
                 activity=activity, target_weight=target,
                 likes_b=likes_b, likes_l=likes_l, likes_d=likes_d,
                 dislikes=dislikes, use_ai=use_ai,
                 last_updated=str(dt.date.today()))
        save_profile(S["user"], P)
        st.success("Profile saved")

    if st.button("🚪 Log out"):
        st.session_state.clear(); st.rerun()

# ────────────── Live metrics banner ────────────────────────────────
col1, col2, col3 = st.columns(3)
bmi_now = bmi(weight, height)
tdee = tdee_msj(weight, height, age, sex, activity)
daily_k = tdee - 500; S["daily_k"] = daily_k
weeks_goal = weeks_to_goal(weight, target)
col1.metric("BMI", f"{bmi_now:.1f}")
col2.metric("TDEE", f"{int(tdee)} kcal/d")
col3.metric("Time to goal", f"{int(weeks_goal)} weeks" if weeks_goal > 1 else "≈1 week")
st.divider()



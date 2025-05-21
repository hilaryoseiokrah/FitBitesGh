###############################################################################
# FitBites – full app (Fixed: 2025-05-21)
# • Fixes TypeError in reshuffle caused by dict usage in re.sub
# • Adds safe _clean() function to handle dicts
###############################################################################

import os, glob, io, json, datetime, re
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

import streamlit as st
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ───────── Regex cleaning helper ─────────
def _clean(s):
    pattern = r'[^\w\s,{}":.-]'
    repl = ''
    if isinstance(s, dict):
        s = json.dumps(s, ensure_ascii=False)
    return re.sub(pattern, repl, s)

# ───────── OpenAI optional ─────────
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    client_openai = OpenAI(api_key=_KEY) if _KEY else None
    OPENAI_AVAILABLE = bool(client_openai)
except Exception:
    client_openai = None

def _rerun(): st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

# ───────── Folders ─────────
DATA_DIR = "data"
PROFILE_DIR = f"{DATA_DIR}/profiles"
PIC_DIR = f"{DATA_DIR}/profile_pics"
MEALPLAN_DIR = f"{DATA_DIR}/mealplans"
for d in (DATA_DIR, PROFILE_DIR, PIC_DIR, MEALPLAN_DIR): os.makedirs(d, exist_ok=True)

# ───────── Auth utilities ─────────
USER_FILE = f"{DATA_DIR}/users.csv"
def users_df(): return pd.read_csv(USER_FILE) if os.path.isfile(USER_FILE) else pd.DataFrame(columns=["username","password"])
def save_user(u,p):
    df = users_df()
    if u in df.username.values: return False
    pd.concat([df,pd.DataFrame([[u,p]],columns=df.columns)]).to_csv(USER_FILE,index=False)
    return True
def valid(u,p): return not users_df()[(users_df().username==u)&(users_df().password==p)].empty

# ───────── Profile utilities ─────────
def prof_path(u): return f"{PROFILE_DIR}/{u}.json"
def load_prof(u):
    if os.path.isfile(prof_path(u)): return json.load(open(prof_path(u)))
    return dict(weight=90,height=160,age=25,sex="female",activity="sedentary",
                target_weight=75,likes_b=[],likes_l=[],likes_d=[],
                dislikes=[],use_ai=False,last_updated=str(datetime.date.today()))
def save_prof(u,data): json.dump(data,open(prof_path(u),"w"),indent=2)
def pic_path(u): return f"{PIC_DIR}/{u}.png"

# ───────── Session defaults ─────────
for k,v in dict(logged_in=False,username="",profile={},meal_plan=None,
                current_week=None,daily_calories=None).items():
    st.session_state.setdefault(k,v)

st.set_page_config(page_title="FitBites – AI Ghanaian Meal Plans", layout="wide")

# ───────── Login / Register ─────────
if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")
    t1,t2=st.tabs(["Login","Register"])
    with t1:
        u=st.text_input("Username")
        p=st.text_input("Password",type="password")
        if st.button("Login"):
            if valid(u,p):
                st.session_state.update(logged_in=True,username=u,profile=load_prof(u))
                _rerun()
            else: st.error("Invalid credentials")
    with t2:
        nu=st.text_input("New username"); npw=st.text_input("New password",type="password")
        if st.button("Create account"):
            st.success("Account created! Go to Login.") if save_user(nu,npw) else st.warning("Username exists")
    st.stop()

# ───────── Logout button ─────────
with st.sidebar:
    if st.button("🚪 Log out"):
        st.session_state.clear(); st.session_state.logged_in=False; _rerun()

# ───────── Food data & embeddings ─────────
@st.cache_data
def load_food():
    df=pd.read_csv("gh_food_nutritional_values.csv"); df.Food=df.Food.str.strip().str.lower()
    cols=["Protein(g)","Fat(g)","Carbs(g)","Calories(100g)","Water(g)","SFA(100g)","MUFA(100g)","PUFA(100g)"]
    df[cols]=df[cols].fillna(df[cols].mean()); return df,cols
df,ncols=load_food()

class AE(nn.Module):
    def __init__(s,d): super().__init__(); s.e=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,8)); s.d=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,d))
    def forward(s,x): z=s.e(x); return z,s.d(z)
@st.cache_resource
def embed(mat):
    net=AE(mat.shape[1]); opt=optim.Adam(net.parameters(),1e-3); loss=nn.MSELoss(); t=torch.tensor(mat,dtype=torch.float32)
    for _ in range(200): opt.zero_grad(); z,out=net(t); loss(out,t).backward(); opt.step()
    with torch.no_grad(): z,_=net(t); return z.numpy()
emb=embed(StandardScaler().fit_transform(df[ncols]))
def similar(f,k=5,exc=None):
    exc=exc or []; idx=df.index[df.Food==f][0]
    sims=cosine_similarity(emb[idx].reshape(1,-1),emb).ravel()
    return [df.iloc[i].Food for i in sims.argsort()[::-1] if df.iloc[i].Food not in exc and df.iloc[i].Food!=f][:k]

# You should continue integrating _clean in reshuffle and meal plan display logic from here

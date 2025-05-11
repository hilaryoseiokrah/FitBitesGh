################################################################################
# FitBites – Weekly Meal-Plan Edition  (2025-05-11)
################################################################################
# • CSV login / register
# • Safe reruns (Streamlit ≥1.34)
# • Welcome banner + Twi proverb + quick guide
# • 7-day Ghanaian meal plan   ─ likes / dislikes
# • “Generate Next Week Plan”, week history viewer, CSV download
# • Partial / full reshuffle on any week
################################################################################

import os, glob, io
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"         # avoid torch watcher crash

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------- helper for old/new rerun ----------
def _safe_rerun():
    if hasattr(st, "rerun"):  st.rerun()
    else:                     st.experimental_rerun()
# ----------------------------------------------

st.set_page_config(page_title="FitBites – Personalized Meal Plans 🇬🇭",
                   layout="wide")

# ────────────────────────── auth helpers ─────────────────────────────────────
USER_FILE = "users.csv"
def load_users():
    return pd.read_csv(USER_FILE) if os.path.exists(USER_FILE) else pd.DataFrame(columns=["username","password"])
def save_user(u,p):
    users = load_users()
    if u in users.username.values: return False
    pd.concat([users, pd.DataFrame([[u,p]], columns=["username","password"])]
             ).to_csv(USER_FILE, index=False)
    return True
def valid_login(u,p):
    users = load_users()
    return not users[(users.username==u) & (users.password==p)].empty
# ────────────────────────── session defaults ────────────────────────────────
for k,v in dict(logged_in=False, username="", meal_plan=None,
                reshuffle_mode=False, daily_calories=None,
                current_week=None).items():
    st.session_state.setdefault(k,v)

# ────────────────────────── LOGIN / REGISTER UI ─────────────────────────────
if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")
    login_tab, reg_tab = st.tabs(["Login", "Register"])

    with login_tab:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if valid_login(u,p):
                st.session_state.logged_in=True
                st.session_state.username=u
                _safe_rerun()
            else:
                st.error("❌ Invalid credentials")

    with reg_tab:
        nu  = st.text_input("Choose Username", key="reg_user")
        npw = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register", key="reg_btn"):
            if save_user(nu,npw):
                st.success("✅ Registered! Switch to Login tab.")
            else:
                st.warning("Username already exists")
    st.stop()

# ────────────────────────── LOG OUT ─────────────────────────────────────────
with st.sidebar:
    if st.button("🚪 Log Out"):
        for k in list(st.session_state.keys()):
            if k!="logged_in": st.session_state.pop(k,None)
        st.session_state.logged_in=False
        _safe_rerun()

# ────────────────────────── WELCOME + GUIDE ────────────────────────────────
st.markdown(f"""
### 👋 Wo ho te sɛn, **{st.session_state.username.title()}**?

🎉 **Welcome to FitBites!**

> *Adidie pa yɛ ahoɔden pa* — **Good food equals good health** 🇬🇭

#### 📝 How to use FitBites
1. Fill in your details on the left.  
2. *(Optional)* choose foods you **like** and foods to **avoid**.  
3. Click **✨ Generate Plan** (or **➕ Generate Next Week Plan**) for a new week.  
4. Use **🔄 Reshuffle Plan** to tweak specific meals or the whole week.  
5. Click **⬇️ Download this plan** to save the current week’s CSV.  
---
""", unsafe_allow_html=True)

# ────────────────────────── FOOD DATA + EMBEDDINGS (FIXED) ──────────────────
@st.cache_data
def load_food():
    df = pd.read_csv("gh_food_nutritional_values.csv")
    df.Food = df.Food.str.strip().str.lower()
    cols = ["Protein(g)","Fat(g)","Carbs(g)","Calories(100g)",
            "Water(g)","SFA(100g)","MUFA(100g)","PUFA(100g)"]
    df[cols] = df[cols].fillna(df[cols].mean())
    return df, cols
df, nut_cols = load_food()

class AE(nn.Module):
    """Simple autoencoder (emb_dim=8 = original features)."""
    def __init__(self, dim, emb_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, emb_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 32), nn.ReLU(), nn.Linear(32, dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

@st.cache_resource
def get_embeddings(matrix):
    net = AE(matrix.shape[1])          # emb_dim = 8
    opt = optim.Adam(net.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    t = torch.tensor(matrix, dtype=torch.float32)
    for _ in range(200):
        opt.zero_grad()
        z, out = net(t)
        crit(out, t).backward()
        opt.step()
    with torch.no_grad():
        z, _ = net(t)
        return z.numpy()

scaled = StandardScaler().fit_transform(df[nut_cols])
emb   = get_embeddings(scaled)

def similar(food, k=5, exclude=None):
    exclude = exclude or []
    idx = df.index[df.Food == food][0]
    sims = cosine_similarity(emb[idx].reshape(1,-1), emb).ravel()
    order = sims.argsort()[::-1]
    return [df.iloc[i].Food for i in order
            if df.iloc[i].Food not in exclude and df.iloc[i].Food != food][:k]

# ────────────────────────── PLAN UTILITIES ─────────────────────────────────
def calc_tdee(w,h,a,sex,act):
    bmr = 10*w + 6.25*h - 5*a + (5 if sex=="male" else -161)
    mult = dict(sedentary=1.2, light=1.375, moderate=1.55,
                active=1.725, superactive=1.9)
    return bmr * mult[act]

def build_plan(prefs, kcal, dislikes):
    split = dict(breakfast=0.25, lunch=0.35, dinner=0.4)
    rows=[]
    for d in range(1,8):
        row={"Day":f"Day {d}"}; tot=0
        for meal,frac in split.items():
            opts=[]
            for s in prefs.get(meal,[]):
                opts += similar(s, exclude=dislikes)
            if not opts:
                opts = list(set(df.Food.sample(5)) - set(dislikes))
            pick = np.random.choice(opts)
            cal100 = df.loc[df.Food==pick,"Calories(100g)"].iat[0]
            grams  = kcal * frac / cal100 * 100
            row[meal.capitalize()] = f"{pick} ({grams:.0f}g)"
            tot += grams
        row["Total Portion (g)"] = f"{tot:.0f}g"; rows.append(row)
    return pd.DataFrame(rows)

def plan_path(user,week): return f"mealplans_{user}_week{week}.csv"
def list_weeks(user):
    files = sorted(glob.glob(f"mealplans_{user}_week*.csv"))
    return [int(f.split('week')[-1].split('.csv')[0]) for f in files]

# ────────────────────────── SIDEBAR PREFS ──────────────────────────────────
with st.sidebar:
    st.subheader("📋 Details")
    w = st.number_input("Weight (kg)",30,200,90)
    h = st.number_input("Height (cm)",120,250,160)
    age = st.number_input("Age",10,100,25)
    sex = st.selectbox("Sex",["female","male"])
    act = st.selectbox("Activity",["sedentary","light","moderate","active","superactive"])

    st.subheader("🍽 Likes (optional)")
    likes_b = st.multiselect("Breakfast",df.Food.unique())
    likes_l = st.multiselect("Lunch",df.Food.unique())
    likes_d = st.multiselect("Dinner",df.Food.unique())

    st.subheader("🚫 Dislikes")
    dislikes = st.multiselect("Never include",df.Food.unique())

    # (Re)Generate current week
    if st.button("✨ Generate Plan"):
        st.session_state.daily_calories = calc_tdee(w,h,age,sex,act) - 500
        prefs = dict(breakfast=likes_b, lunch=likes_l, dinner=likes_d)
        week  = st.session_state.current_week or (max(list_weeks(st.session_state.username))+1 if list_weeks(st.session_state.username) else 1)
        plan  = build_plan(prefs, st.session_state.daily_calories, dislikes)
        plan.to_csv(plan_path(st.session_state.username, week), index=False)
        st.session_state.meal_plan = plan
        st.session_state.current_week = week
        st.session_state.reshuffle_mode=False
        st.success(f"Week {week} generated & saved")

    # Generate NEXT week
    if st.button("➕ Generate Next Week Plan"):
        st.session_state.daily_calories = calc_tdee(w,h,age,sex,act) - 500
        prefs = dict(breakfast=likes_b, lunch=likes_l, dinner=likes_d)
        next_week = (max(list_weeks(st.session_state.username))+1) if list_weeks(st.session_state.username) else 1
        plan = build_plan(prefs, st.session_state.daily_calories, dislikes)
        plan.to_csv(plan_path(st.session_state.username,next_week), index=False)
        st.session_state.meal_plan = plan
        st.session_state.current_week = next_week
        st.session_state.reshuffle_mode=False
        st.success(f"Week {next_week} generated & saved")

# ────────────────────────── Week picker ─────────────────────────────────────
weeks_avail = list_weeks(st.session_state.username)
if weeks_avail:
    sel_week = st.selectbox("📆 View week", weeks_avail,
                            index=weeks_avail.index(st.session_state.current_week or weeks_avail[-1]))
    if sel_week != st.session_state.current_week:
        st.session_state.current_week = sel_week
        st.session_state.meal_plan   = pd.read_csv(plan_path(st.session_state.username, sel_week))
        st.session_state.reshuffle_mode=False

# ────────────────────────── Display + download ─────────────────────────────
if st.session_state.meal_plan is not None:
    st.subheader(f"📅 Week {st.session_state.current_week} Meal Plan")
    st.dataframe(st.session_state.meal_plan,use_container_width=True)
    st.download_button("⬇️ Download this plan",
                       st.session_state.meal_plan.to_csv(index=False).encode(),
                       file_name=f"mealplan_week{st.session_state.current_week}.csv",
                       mime="text/csv")
    if st.button("🔄 Reshuffle Plan"): st.session_state.reshuffle_mode=True

# ────────────────────────── Reshuffle UI/logic ─────────────────────────────
if st.session_state.reshuffle_mode and st.session_state.meal_plan is not None:
    st.markdown("---"); st.markdown("### 🔄 Reshuffle Options")
    mode = st.radio("Choose",["Partial","Full"], horizontal=True)
    prefs = dict(breakfast=likes_b, lunch=likes_l, dinner=likes_d)

    if mode=="Partial":
        days_sel = st.multiselect("Days", st.session_state.meal_plan.Day.tolist())
        meals_sel= st.multiselect("Meals",["Breakfast","Lunch","Dinner"])
        extra_dis = st.multiselect("Extra dislikes for this reshuffle", df.Food.unique())
        if st.button("Apply Partial Reshuffle"):
            upd_dis = list(set(dislikes+extra_dis))
            new = build_plan(prefs, st.session_state.daily_calories, upd_dis)
            for day in days_sel:
                old_i = st.session_state.meal_plan.index[st.session_state.meal_plan.Day==day][0]
                new_i = new.index[new.Day==day][0]
                for m in meals_sel:
                    st.session_state.meal_plan.at[old_i,m] = new.at[new_i,m]
                st.session_state.meal_plan.at[old_i,"Total Portion (g)"] = new.at[new_i,"Total Portion (g)"]
            st.session_state.meal_plan.to_csv(plan_path(st.session_state.username, st.session_state.current_week), index=False)
            st.session_state.reshuffle_mode=False
            _safe_rerun()

    if mode=="Full":
        extra_dis_f = st.multiselect("Extra dislikes for NEW plan", df.Food.unique())
        if st.button("Apply Full Reshuffle"):
            upd_dis = list(set(dislikes+extra_dis_f))
            st.session_state.meal_plan = build_plan(prefs, st.session_state.daily_calories, upd_dis)
            st.session_state.meal_plan.to_csv(plan_path(st.session_state.username, st.session_state.current_week), index=False)
            st.session_state.reshuffle_mode=False
            _safe_rerun()

###############################################################################
# FitBites – full app  (2025-05-12)
# • Login / Register
# • Profile page (info + picture + plan history)
# • Weekly meal plans (AI or classic)  + Reshuffles + CSV download
# • AI Recipe Maker
###############################################################################

import os, glob, io, json, datetime
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

import streamlit as st
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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
DATA_DIR      = "data"
PROFILE_DIR   = f"{DATA_DIR}/profiles"
PIC_DIR       = f"{DATA_DIR}/profile_pics"
MEALPLAN_DIR  = f"{DATA_DIR}/mealplans"
for d in (DATA_DIR, PROFILE_DIR, PIC_DIR, MEALPLAN_DIR): os.makedirs(d, exist_ok=True)

# ───────── Auth utilities ─────────
USER_FILE = f"{DATA_DIR}/users.csv"
def users_df(): return pd.read_csv(USER_FILE) if os.path.isfile(USER_FILE) else pd.DataFrame(columns=["username","password"])
def save_user(u,p):
    df=users_df()
    if u in df.username.values: return False
    pd.concat([df,pd.DataFrame([[u,p]],columns=df.columns)]).to_csv(USER_FILE,index=False); return True
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

# ───────── Plan builders & recipe LLM ─────────
def calc_tdee(w,h,a,sex,act):
    mult=dict(sedentary=1.2,light=1.375,moderate=1.55,active=1.725,superactive=1.9)[act]
    return (10*w+6.25*h-5*a+(5 if sex=="male" else -161))*mult
def classic_plan(prefs,kcal,dislikes):
    rows=[]; split=dict(breakfast=0.25,lunch=0.35,dinner=0.4)
    for d in range(1,8):
        row={"Day":f"Day {d}"}; tot=0
        for meal,f in split.items():
            opts=[]; [opts.extend(similar(s,exc=dislikes)) for s in prefs.get(meal,[])]
            if not opts: opts=list(set(df.Food.sample(5))-set(dislikes))
            pick=np.random.choice(opts); cal100=df.loc[df.Food==pick,"Calories(100g)"].iat[0]
            g=kcal*f/cal100*100; row[meal.capitalize()]=f"{pick} ({g:.0f}g)"; tot+=g
        row["Total Portion (g)"]=f"{tot:.0f}g"; rows.append(row)
    return pd.DataFrame(rows)
def gpt_plan(prefs,dislikes,kcal):
    if not OPENAI_AVAILABLE: return None
    likes=", ".join(set(sum(prefs.values(),[]))) or "any Ghanaian foods"
    dis=", ".join(dislikes) if dislikes else "none"
    prompt=f"You are a Ghanaian dietitian. Make 7-day JSON table (Day, Breakfast, Lunch, Dinner) using Ghanaian household measures. Daily ≈{int(kcal)} kcal (25/35/40). LIKES:{likes}. DISLIKES:{dis}. Include kcal in parentheses."
    try:
        r=client_openai.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt}],temperature=0.7,timeout=30)
        return pd.read_json(io.StringIO(r.choices[0].message.content.strip()))
    except Exception as e: st.error(e); return None
def recipe_llm(ing,cui):
    if not OPENAI_AVAILABLE: return None
    sys="You are a recipe dictionary. Only respond with recipes ..."
    user=f"Ingredients: {ing}. Cuisine: {cui}."
    try:
        r=client_openai.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.7,timeout=30)
        return r.choices[0].message.content.strip()
    except Exception as e: st.error(e); return None

# ───────── File helpers ─────────
def ppath(u,w): return f"{MEALPLAN_DIR}/{u}_week{w}.csv"
def weeks(u): return [int(f.split("week")[-1].split(".")[0]) for f in glob.glob(f"{MEALPLAN_DIR}/{u}_week*.csv")]

# ───────── Tabs ─────────
tab_plan, tab_profile, tab_recipe = st.tabs(["🍽️ Planner","👤 Profile","🍲 Recipe"])

# ===== Planner =====
with tab_plan:
    st.header("Weekly Meal Planner")
    prof=st.session_state.profile
    with st.sidebar:
        st.subheader("Details")
        w = st.number_input("Weight kg", 30.0, 200.0, float(prof["weight"]), step=0.1)
        h = st.number_input("Height cm", 120.0, 250.0, float(prof["height"]), step=0.1)
        age = st.number_input("Age", 10, 100, int(prof["age"]), step=1)
        sex = st.selectbox("Sex",["female","male"],index=0 if prof["sex"]=="female" else 1)
        act = st.selectbox("Activity",["sedentary","light","moderate","active","superactive"],
                           index=["sedentary","light","moderate","active","superactive"].index(prof["activity"]))

        st.subheader("Likes")
        likes_b = st.multiselect("Breakfast", df.Food.unique(), default=prof.get("likes_b",[]))
        likes_l = st.multiselect("Lunch",     df.Food.unique(), default=prof.get("likes_l",[]))
        likes_d = st.multiselect("Dinner",    df.Food.unique(), default=prof.get("likes_d",[]))

        st.subheader("Dislikes")
        dislikes = st.multiselect("Dislikes", df.Food.unique(), default=prof.get("dislikes",[]))

        use_ai = st.checkbox("🤖 AI combos", value=prof.get("use_ai",False), disabled=not OPENAI_AVAILABLE)

        def save_plan(for_next=False):
            st.session_state.daily_calories = calc_tdee(w,h,age,sex,act)-500
            prefs=dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
            dis=dislikes
            week = (max(weeks(st.session_state.username))+1 if weeks(st.session_state.username) else 1) if for_next else \
                   (st.session_state.current_week or 1)
            plan = gpt_plan(prefs,dis,st.session_state.daily_calories) if use_ai else classic_plan(prefs,st.session_state.daily_calories,dis)
            if plan is not None:
                plan.to_csv(ppath(st.session_state.username,week),index=False)
                st.session_state.meal_plan, st.session_state.current_week = plan, week
                st.success(f"Week {week} saved")
            prof.update(dict(weight=w,height=h,age=age,sex=sex,activity=act,
                             likes_b=likes_b,likes_l=likes_l,likes_d=likes_d,
                             dislikes=dis,use_ai=use_ai,last_updated=str(datetime.date.today())))
            save_prof(st.session_state.username,prof)
            st.session_state.profile=prof

        if st.button("✨ Generate / Update plan"): save_plan()
        if st.button("➕ Next week"): save_plan(for_next=True)

    if st.session_state.meal_plan is not None:
        st.selectbox("📆 Week",weeks(st.session_state.username),
                     index=weeks(st.session_state.username).index(st.session_state.current_week),
                     key="week_sel",on_change=lambda: st.session_state.update(
                         meal_plan=pd.read_csv(ppath(st.session_state.username,st.session_state.week_sel)),
                         current_week=st.session_state.week_sel))
        st.dataframe(st.session_state.meal_plan, use_container_width=True)
        st.download_button("⬇️ CSV",st.session_state.meal_plan.to_csv(index=False).encode(),
                           file_name=f"mealplan_week{st.session_state.current_week}.csv")

# ===== Profile =====
with tab_profile:
    st.header("Profile")
    col1,col2=st.columns([1,3])
    with col1:
        if os.path.isfile(pic_path(st.session_state.username)):
            st.image(pic_path(st.session_state.username),width=150)
        up=st.file_uploader("Upload photo",type=["png","jpg","jpeg"])
        if up:
            open(pic_path(st.session_state.username),"wb").write(up.getbuffer())
            st.success("Saved. Refresh.")
    with col2:
        p=st.session_state.profile
        st.markdown(f"""
* **Weight:** {p['weight']} kg  
* **Height:** {p['height']} cm  
* **Target weight:** {p['target_weight']} kg  
* **Age:** {p['age']}  
* **Sex:** {p['sex'].title()}  
* **Activity:** {p['activity'].title()}  
* **Last updated:** {p['last_updated']}
""")
    st.divider(); st.subheader("Saved plans")
    for w in weeks(st.session_state.username):
        with st.expander(f"Week {w}"):
            st.download_button("Download",open(ppath(st.session_state.username,w),"rb").read(),
                               file_name=f"mealplan_week{w}.csv",key=f"d{w}")

# ===== Recipe =====
with tab_recipe:
    st.header("AI Recipe Maker")
    ing=st.text_area("Ingredients (comma-separated)")
    cui=st.text_input("Cuisine")
    if st.button("Generate recipe"):
        if not ing.strip() or not cui.strip(): st.warning("Fill both fields.")
        else:
            with st.spinner("GPT cooking…"):
                r=recipe_llm(ing,cui)
            if r:
                st.markdown(r); st.download_button("Save txt",r.encode(),
                                                   file_name="recipe.txt",mime="text/plain")

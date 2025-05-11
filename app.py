###############################################################################
# FitBites – v2025-05-12  (Profile-tab fixed: consistent number_input types)
###############################################################################

import os, glob, io, json, time, datetime
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

import streamlit as st
import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------- OpenAI optional ----------
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    client_openai = OpenAI(api_key=_key) if _key else None
    OPENAI_AVAILABLE = bool(client_openai)
except Exception:
    client_openai = None

def _safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    else:                    st.experimental_rerun()

st.set_page_config(page_title="FitBites – AI Ghanaian Plans", layout="wide")

# ----- folders -----
DATA_DIR, PROFILE_DIR, PIC_DIR, MEALPLAN_DIR = "data", "data/profiles", "data/profile_pics", "data/mealplans"
for p in (DATA_DIR, PROFILE_DIR, PIC_DIR, MEALPLAN_DIR): os.makedirs(p, exist_ok=True)

# ─────────────────── auth utils ───────────────────
USER_FILE = os.path.join(DATA_DIR, "users.csv")
def load_users():
    return pd.read_csv(USER_FILE) if os.path.exists(USER_FILE) else pd.DataFrame(columns=["username","password"])
def save_user(u,p):
    users = load_users()
    if u in users.username.values: return False
    pd.concat([users,pd.DataFrame([[u,p]],columns=["username","password"])]
             ).to_csv(USER_FILE,index=False); return True
def valid_login(u,p): return not load_users()[(load_users().username==u)&(load_users().password==p)].empty

# ─────────────────── profile io ───────────────────
def profile_path(user): return f"{PROFILE_DIR}/{user}.json"
def load_profile(user):
    if os.path.isfile(profile_path(user)):
        return json.load(open(profile_path(user)))
    return dict(weight=90,height=160,age=25,sex="female",activity="sedentary",
                target_weight=75,last_updated=str(datetime.date.today()))
def save_profile(user,data): json.dump(data,open(profile_path(user),"w"),indent=2)
def pic_path(user): return f"{PIC_DIR}/{user}.png"

# ─────────────────── session defaults ─────────────
for k,v in dict(logged_in=False,username="",meal_plan=None,current_week=None,
                reshuffle_mode=False,daily_calories=None).items():
    st.session_state.setdefault(k,v)

# ─────────────────── LOGIN / REGISTER ─────────────
if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")
    t1,t2=st.tabs(["Login","Register"])
    with t1:
        u=st.text_input("Username"); p=st.text_input("Password",type="password")
        if st.button("Login"):
            if valid_login(u,p):
                st.session_state.logged_in=True; st.session_state.username=u
                st.session_state.profile=load_profile(u); _safe_rerun()
            else: st.error("❌ Invalid credentials")
    with t2:
        nu=st.text_input("Choose Username",key="regu"); npw=st.text_input("Choose Password",type="password",key="regp")
        if st.button("Register",key="regbtn"):
            st.success("✅ Registered!") if save_user(nu,npw) else st.warning("User exists")
    st.stop()

# ─────────────────── LOGOUT ───────────────────────
with st.sidebar:
    if st.button("🚪 Log Out"):
        for k in list(st.session_state.keys()):
            if k!="logged_in": st.session_state.pop(k,None)
        st.session_state.logged_in=False; _safe_rerun()

# ─────────────────── Food data + embeddings ───────
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
def similar(food,k=5,exc=None):
    exc=exc or []; idx=df.index[df.Food==food][0]; sims=cosine_similarity(emb[idx].reshape(1,-1),emb).ravel()
    return [df.iloc[i].Food for i in sims.argsort()[::-1] if df.iloc[i].Food not in exc and df.iloc[i].Food!=food][:k]

# ─────────────────── planners / recipe LLM ───────
def calc_tdee(w,h,a,sex,act):
    mult=dict(sedentary=1.2,light=1.375,moderate=1.55,active=1.725,superactive=1.9)[act]
    return (10*w+6.25*h-5*a+(5 if sex=="male" else -161))*mult
def build_plan(prefs,kcal,dislikes):
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
    prompt=f"""
You are a Ghanaian dietitian. Build a 7-day JSON table with keys Day, Breakfast, Lunch, Dinner.
Use household measures, include kcal in parentheses. Daily ≈ {int(kcal)} kcal (25/35/40).
LIKES: {likes}. DISLIKES: {dis}."""
    try:
        r=client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,timeout=30)
        return pd.read_json(io.StringIO(r.choices[0].message.content.strip()))
    except Exception as e: st.error(e); return None
def recipe_llm(ing,cui):
    if not OPENAI_AVAILABLE: return None
    sys="You are a recipe dictionary. Only respond with recipes..."
    user=f"Ingredients: {ing}. Cuisine: {cui}."
    try:
        r=client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.7,timeout=30)
        return r.choices[0].message.content.strip()
    except Exception as e: st.error(e); return None

# ─────────────────── helper files ────────────────
def plan_path(u,w): return f"{MEALPLAN_DIR}/{u}_week{w}.csv"
def week_list(u): return [int(f.split("week")[-1].split(".")[0]) for f in glob.glob(f"{MEALPLAN_DIR}/{u}_week*.csv")]

# ─────────────────── TABS ─────────────────────────
tab_plan,tab_profile,tab_recipe=st.tabs(["🍽️ Meal Planner","👤 Profile","🍲 Recipe Maker"])

# ----------------- Meal planner ------------------
with tab_plan:
    st.header("Weekly Meal Planner")
    prof=st.session_state.get("profile",load_profile(st.session_state.username))
    with st.sidebar:
        st.subheader("📋 Details")
        w=st.number_input("Weight (kg)",min_value=30.0,max_value=200.0,value=float(prof["weight"]),step=0.1)
        h=st.number_input("Height (cm)",min_value=120.0,max_value=250.0,value=float(prof["height"]),step=0.1)
        age=int(st.number_input("Age",min_value=10,max_value=100,value=int(prof["age"]),step=1))
        sex=st.selectbox("Sex",["female","male"],index=0 if prof["sex"]=="female" else 1)
        act=st.selectbox("Activity",["sedentary","light","moderate","active","superactive"],
                         index=["sedentary","light","moderate","active","superactive"].index(prof["activity"]))

        st.subheader("🍽 Likes")
        likes_b=st.text_input("Breakfast likes (comma)",", ".join(prof.get("likes_b",[])))
        likes_l=st.text_input("Lunch likes (comma)",", ".join(prof.get("likes_l",[])))
        likes_d=st.text_input("Dinner likes (comma)",", ".join(prof.get("likes_d",[])))

        st.subheader("🚫 Dislikes")
        dislikes_txt=st.text_input("Dislikes (comma)",", ".join(prof.get("dislikes",[])))

        use_ai=st.checkbox("🤖 AI-generated combos",value=prof.get("use_ai",False),disabled=not OPENAI_AVAILABLE)

        if st.button("✨ Generate Plan"):
            st.session_state.daily_calories=calc_tdee(w,h,age,sex,act)-500
            prefs=dict(breakfast=[x.strip() for x in likes_b.split(",") if x.strip()],
                       lunch=[x.strip() for x in likes_l.split(",") if x.strip()],
                       dinner=[x.strip() for x in likes_d.split(",") if x.strip()])
            dis=[x.strip() for x in dislikes_txt.split(",") if x.strip()]
            week=st.session_state.current_week or (max(week_list(st.session_state.username))+1 if week_list(st.session_state.username) else 1)
            plan=gpt_plan(prefs,dis,st.session_state.daily_calories) if use_ai else build_plan(prefs,st.session_state.daily_calories,dis)
            if plan is not None:
                plan.to_csv(plan_path(st.session_state.username,week),index=False)
                st.session_state.meal_plan=plan; st.session_state.current_week=week
            prof.update(dict(weight=w,height=h,age=age,sex=sex,activity=act,
                             likes_b=prefs["breakfast"],likes_l=prefs["lunch"],likes_d=prefs["dinner"],
                             dislikes=dis,use_ai=use_ai,last_updated=str(datetime.date.today())))
            save_profile(st.session_state.username,prof)
            st.session_state.profile=prof

    weeks=week_list(st.session_state.username)
    if weeks:
        sel=st.selectbox("📆 Week",weeks,index=weeks.index(st.session_state.current_week or weeks[-1]))
        if sel!=st.session_state.current_week:
            st.session_state.current_week=sel
            st.session_state.meal_plan=pd.read_csv(plan_path(st.session_state.username,sel))
    if st.session_state.meal_plan is not None:
        st.subheader(f"Week {st.session_state.current_week}")
        st.dataframe(st.session_state.meal_plan,use_container_width=True)
        st.download_button("⬇️ CSV",st.session_state.meal_plan.to_csv(index=False).encode(),
                           file_name=f"mealplan_week{st.session_state.current_week}.csv")

# ----------------- Profile -----------------------
with tab_profile:
    st.header("👤 Profile")
    prof=st.session_state.get("profile",load_profile(st.session_state.username))
    col1,col2=st.columns([1,3])
    with col1:
        if os.path.isfile(pic_path(st.session_state.username)):
            st.image(pic_path(st.session_state.username),width=150)
        up=st.file_uploader("Upload photo",type=["png","jpg","jpeg"])
        if up: open(pic_path(st.session_state.username),"wb").write(up.getbuffer()); st.success("Saved.")
    with col2:
        st.markdown(f"""
* **Weight:** {prof['weight']} kg  
* **Height:** {prof['height']} cm  
* **Target weight:** {prof.get('target_weight','?')} kg  
* **Age:** {prof['age']}  
* **Sex:** {prof['sex'].title()}  
* **Activity:** {prof['activity'].title()}  
* **Last update:** {prof.get('last_updated','-')}
""")
    st.divider(); st.subheader("📚 Saved plans")
    for w in week_list(st.session_state.username):
        colA,colB=st.columns([3,1])
        with colA: st.write(f"Week {w}")
        with colB:
            with open(plan_path(st.session_state.username,w),"rb") as f:
                st.download_button("Download",f.read(),file_name=f"mealplan_week{w}.csv",key=f"d{w}")

# ----------------- Recipe maker ------------------
with tab_recipe:
    st.header("🍲 Recipe Maker")
    ing=st.text_area("Ingredients (comma-separated)")
    cui=st.text_input("Cuisine")
    if st.button("Generate recipe"):
        if not ing.strip() or not cui.strip(): st.warning("Enter both.")
        else:
            with st.spinner("GPT cooking…"):
                rec=recipe_llm(ing,cui)
            if rec:
                st.markdown(rec); st.download_button("⬇️ Save",rec.encode(),"recipe.txt","text/plain")

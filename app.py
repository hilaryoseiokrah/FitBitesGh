###############################################################################
#  FitBites – Ghana-centric AI Meal-Planner
#  Rev. 2025-05-21  •  secure auth + evidence-based nutrition
###############################################################################
import os, glob, json, datetime as dt, re, random
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from passlib.hash import bcrypt            # pip install passlib[bcrypt]

# ───────────────────────── OpenAI (optional) ────────────────────────────────
try:
    from openai import OpenAI
    client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY",
                                             st.secrets.get("OPENAI_API_KEY","")))
    OPENAI_AVAILABLE = True
except Exception:
    client_openai, OPENAI_AVAILABLE = None, False

# ─────────────────────────────── Paths ──────────────────────────────────────
BASE      = Path.cwd()
DATA      = BASE / "data"
PROFILES  = DATA / "profiles"
PICS      = DATA / "profile_pics"
PLANS     = DATA / "mealplans"
for p in (DATA, PROFILES, PICS, PLANS): p.mkdir(parents=True, exist_ok=True)

USERS_CSV = DATA / "users.csv"

# ───────────────────────── Secure authentication ───────────────────────────
def _users_df() -> pd.DataFrame:
    if USERS_CSV.exists():
        return pd.read_csv(USERS_CSV)
    return pd.DataFrame(columns=["username","pwd_hash"])

def register_user(username:str, password:str) -> bool:
    df = _users_df()
    if username in df.username.values:
        return False
    df.loc[len(df)] = [username, bcrypt.hash(password)]
    df.to_csv(USERS_CSV, index=False)
    return True

def authenticate(username:str, password:str) -> bool:
    df = _users_df()
    row = df[df.username == username]
    if row.empty:                       # user not found
        return False
    return bcrypt.verify(password, row.iloc[0].pwd_hash)

# ────────────────────────── Profile IO helpers ─────────────────────────────
def profile_path(u):   return PROFILES / f"{u}.json"
def pic_path(u):       return PICS / f"{u}.png"
def plan_path(u,w):    return PLANS / f"{u}_week{w}.csv"

def load_profile(u:str) -> dict:
    if profile_path(u).exists():
        return json.loads(profile_path(u).read_text())
    return dict(weight=90,height=160,age=25,sex="female",activity="sedentary",
                target_weight=75,likes_b=[],likes_l=[],likes_d=[],dislikes=[],
                use_ai=False,last_updated=str(dt.date.today()))

def save_profile(u:str, d:dict)->None:
    profile_path(u).write_text(json.dumps(d, indent=2))

def existing_weeks(u:str) -> list[int]:
    w = [re.search(r"_week(\d+)", f.name) for f in PLANS.glob(f"{u}_week*.csv")]
    return sorted(int(m.group(1)) for m in w if m)

# ──────────────────────────── BMI • TDEE • Goal time ───────────────────────
def bmi(w:float,h_cm:float)->float: return w/(h_cm/100)**2

def tdee_msj(w,h_cm,age,sex,activity)->float:
    base = 10*w + 6.25*h_cm - 5*age + (5 if sex=="male" else -161)
    factors = dict(sedentary=1.2,light=1.375,moderate=1.55,active=1.725,superactive=1.9)
    return base * factors[activity]

SAFE_KG_PER_WEEK = 0.75  # CDC: 0.45–0.9 kg (1-2 lb) → choose mid-point

def weeks_to_goal(cur,target): return abs(cur-target)/SAFE_KG_PER_WEEK

# ────────────────────── Data: Ghanaian food + embeddings ───────────────────
@st.cache_data(show_spinner=False)
def load_food():
    df = pd.read_csv("gh_food_nutritional_values.csv")
    df["Food"] = df["Food"].str.strip().str.lower()
    cols = ["Protein(g)","Fat(g)","Carbs(g)","Calories(100g)",
            "Water(g)","SFA(100g)","MUFA(100g)","PUFA(100g)"]
    df[cols] = df[cols].fillna(df[cols].mean())
    return df, cols
FOODS_DF, NUTR_COLS = load_food()

class AutoEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,8))
        self.dec = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self,x): z=self.enc(x); return z, self.dec(z)

@st.cache_resource(show_spinner=False)
def build_embeddings(arr:np.ndarray)->np.ndarray:
    torch.manual_seed(7)
    net=AutoEncoder(arr.shape[1])
    opt=optim.Adam(net.parameters(),1e-3)
    lossf=nn.MSELoss()
    t=torch.tensor(arr,dtype=torch.float32)
    for _ in range(300):
        opt.zero_grad(); z,out=net(t); lossf(out,t).backward(); opt.step()
    with torch.no_grad(): z,_=net(t); return z.numpy()
SCALER = StandardScaler()
EMB    = build_embeddings(SCALER.fit_transform(FOODS_DF[NUTR_COLS]))

def similar(food:str,k=5,exclude=None):
    exclude = exclude or []
    idx = FOODS_DF.index[FOODS_DF.Food==food]
    if idx.empty: return []
    sims = cosine_similarity(EMB[idx[0]].reshape(1,-1), EMB).ravel().argsort()[::-1]
    res=[]
    for i in sims:
        nm=FOODS_DF.iloc[i].Food
        if nm not in exclude and nm!=food:
            res.append(nm)
        if len(res)==k: break
    return res

# ─────────────────────────── Streamlit setup ───────────────────────────────
st.set_page_config("FitBites – AI Ghanaian Meal Plans", layout="wide")

if "state" not in st.session_state:
    st.session_state.state = dict(logged=False,user="",prof={},
                                  plan=None,week=None,daily_k=None)
S = st.session_state.state

# ─────────────────────────────  LOGIN  ─────────────────────────────────────
if not S["logged"]:
    st.title("🔐 FitBites Login")
    t1,t2 = st.tabs(("Login","Register"))
    with t1:
        u = st.text_input("Username")
        p = st.text_input("Password",type="password")
        if st.button("Login"):
            if authenticate(u,p):
                S.update(dict(logged=True,user=u,prof=load_profile(u)))
                st.experimental_rerun()
            else: st.error("Bad credentials")
    with t2:
        u2 = st.text_input("New username")
        p2 = st.text_input("New password",type="password")
        if st.button("Create account"):
            ok = register_user(u2,p2)
            st.success("Account created") if ok else st.warning("User exists")
    st.stop()

# -- quick vars
P=S["prof"]; username=S["user"]

# ─────────── Sidebar – live inputs (updates saved on demand) ───────────────
with st.sidebar:
    st.subheader(f"Hi {username.title()}")
    w  = st.number_input("Weight kg",30.,200.,float(P["weight"]),0.1)
    tw = st.number_input("Target kg",30.,200.,float(P["target_weight"]),0.1)
    h  = st.number_input("Height cm",120.,250.,float(P["height"]),0.1)
    age= st.number_input("Age",10,100,int(P["age"]),1)
    sex= st.selectbox("Sex",["female","male"],0 if P["sex"]=="female" else 1)
    act= st.selectbox("Activity",["sedentary","light","moderate","active","superactive"],
                      ["sedentary","light","moderate","active","superactive"].index(P["activity"]))
    st.markdown("---")
    st.subheader("Food preferences")
    likes_b = st.multiselect("Breakfast likes", FOODS_DF.Food, P["likes_b"])
    likes_l = st.multiselect("Lunch likes"    , FOODS_DF.Food, P["likes_l"])
    likes_d = st.multiselect("Dinner likes"   , FOODS_DF.Food, P["likes_d"])
    dislikes= st.multiselect("Dislikes",         FOODS_DF.Food, P["dislikes"])
    use_ai  = st.checkbox("Use GPT meal-combos", P["use_ai"], disabled=not OPENAI_AVAILABLE)

    if st.button("Save profile"):
        P.update(weight=w,target_weight=tw,height=h,age=age,sex=sex,activity=act,
                 likes_b=likes_b,likes_l=likes_l,likes_d=likes_d,
                 dislikes=dislikes,use_ai=use_ai,last_updated=str(dt.date.today()))
        save_profile(username,P)
        st.success("Profile stored")

    if st.button("Logout"):
        st.session_state.clear(); st.experimental_rerun()

# ───────────── Real-time metrics banner (BMI, TDEE, goal) ───────────────────
col1,col2,col3 = st.columns(3)
BMI    = bmi(w,h);     col1.metric("BMI",f"{BMI:.1f}")
TDEE   = tdee_msj(w,h,age,sex,act); col2.metric("TDEE",f"{int(TDEE)} kcal/d")
dailyk = TDEE - 500;   S["daily_k"] = dailyk
weeks  = weeks_to_goal(w,tw); col3.metric("Time-to-goal",f"{int(weeks)} wk")

st.divider()

# ─────────────── Helper: build 7-day plan (classic or GPT) ──────────────────
def build_classic(kcal:float, prefs:dict, dislikes:list[str])->pd.DataFrame:
    split = dict(breakfast=.25,lunch=.35,dinner=.40)
    rows=[]
    for d in range(1,8):
        row,total={"Day":f"Day {d}"},0
        for meal,f in split.items():
            pool=[]
            for liked in prefs.get(meal,[]):
                pool+=similar(liked,exclude=dislikes)
            if not pool:
                pool=[f for f in FOODS_DF.Food.sample(40) if f not in dislikes]
            pick=random.choice(pool)
            cal100=FOODS_DF.loc[FOODS_DF.Food==pick,"Calories(100g)"].iat[0]
            grams = kcal*f/cal100*100
            row[meal.capitalize()] = f"{pick} ({grams:.0f} g)"
            total += grams
        row["Total g"] = f"{total:.0f}"
        rows.append(row)
    return pd.DataFrame(rows)

def build_gpt(kcal:float,prefs:dict,dislikes:list[str])->pd.DataFrame|None:
    if not OPENAI_AVAILABLE: return None
    likes=", ".join(set(sum(prefs.values(),[]))) or "any Ghanaian foods"
    dis  =", ".join(dislikes) if dislikes else "none"
    sys="You are a registered Ghanaian dietitian. Output JSON only."
    rules=(
        "Pairing rules:\n"
        "• Hausa koko or koko must accompany bofrot or bread & groundnuts\n"
        "• Fufu pairs with any soup except okro (no stews)\n"
        "• Greek yoghurt + banana\n"
        "• Yam or cassava pair with stew; rice → stew; banku / akple → okro soup + protein\n"
        "No imaginary dishes."
    )
    user=(f"{rules}\nCreate 7-day table (Day,Breakfast,Lunch,Dinner). "
          f"Keep ≈{int(kcal)} kcal per day (25/35/40 split). "
          f"LIKES: {likes}.  DISLIKES: {dis}. "
          "Include household measures & kcal per item.")
    try:
        r=client_openai.chat.completions.create(model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.3,timeout=30)
        raw=r.choices[0].message.content
        js=raw[raw.find("["):raw.rfind("]")+1]
        return pd.DataFrame(json.loads(js))
    except Exception as e:
        st.error(f"GPT error: {e}"); return None

def generate_plan(next_week=False):
    prefs=dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
    week=(max(existing_weeks(username) or [0])+1) if next_week else (S.get("week") or 1)
    df= build_gpt(dailyk,prefs,dislikes) if use_ai else build_classic(dailyk,prefs,dislikes)
    if df is not None:
        df.to_csv(plan_path(username,week),index=False)
        S.update(plan=df,week=week)
        st.success(f"Week {week} saved")

# ───────────── Tabs – Planner • Profile • Recipe ────────────────────────────
tab_plan,tab_prof,tab_rec = st.tabs(("🍽 Planner","👤 Profile","🍲 Recipe"))

with tab_plan:
    c1,c2 = st.columns(2)
    if c1.button("Generate / update this week"): generate_plan(next_week=False)
    if c2.button("Generate next-week plan"): generate_plan(next_week=True)

    wks = existing_weeks(username)
    if wks:
        sel = st.selectbox("Week",wks,index=wks.index(S.get("week") or wks[-1]))
        if sel!=S.get("week"):
            S.update(plan=pd.read_csv(plan_path(username,sel)),week=sel)

    if S.get("plan") is not None:
        st.dataframe(S["plan"],use_container_width=True)
        st.download_button("Download CSV",
                           S["plan"].to_csv(index=False).encode(),
                           f"mealplan_week{S['week']}.csv")
        with st.expander("Reshuffle meals"):
            mode=st.radio("Mode",("Partial","Full"),horizontal=True)
            extra_dis=st.multiselect("Extra dislikes",FOODS_DF.Food,[])
            if mode=="Partial":
                days=st.multiselect("Days",S["plan"].Day)
                meals=st.multiselect("Meals",["Breakfast","Lunch","Dinner"])
                if st.button("Apply partial"):
                    prefs=dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
                    new=(build_gpt if use_ai else build_classic)(dailyk,prefs,dislikes+extra_dis)
                    if new is not None:
                        for d in days:
                            io=S["plan"].index[S["plan"].Day==d][0]
                            inw=new.index[new.Day==d][0]
                            for m in meals:
                                S["plan"].at[io,m]=new.at[inw,m]
                            S["plan"].at[io,"Total g"]=new.at[inw,"Total g"]
                        S["plan"].to_csv(plan_path(username,S["week"]),index=False)
                        st.success("Partial reshuffle done")
            else:
                if st.button("Apply full"):
                    prefs=dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
                    S["plan"]=(build_gpt if use_ai else build_classic)(dailyk,prefs,dislikes+extra_dis)
                    S["plan"].to_csv(plan_path(username,S["week"]),index=False)
                    st.success("Full reshuffle done")

with tab_prof:
    st.header("Profile")
    c1,c2=st.columns([1,3])
    with c1:
        if pic_path(username).exists():
            st.image(pic_path(username),width=180)
        up=st.file_uploader("Upload photo",type=["png","jpg","jpeg"])
        if up:
            pic_path(username).write_bytes(up.getbuffer()); st.success("Saved")
    with c2:
        st.markdown(f"""
* **Weight:** {P['weight']} kg  
* **Height:** {P['height']} cm  
* **Target:** {P['target_weight']} kg  
* **Age:** {P['age']}  
* **Sex:** {P['sex']}  
* **Activity:** {P['activity']}  
* **Last update:** {P['last_updated']}
""")

    st.divider(); st.subheader("Saved plans")
    for w in existing_weeks(username):
        with st.expander(f"Week {w}"):
            st.download_button("CSV",open(plan_path(username,w),"rb").read(),
                               f"mealplan_week{w}.csv",key=f"dl{w}")

with tab_rec:
    st.header("Recipe generator")
    if S.get("plan") is None:
        st.info("Generate a plan first")
    else:
        def _strip(txt): return re.sub(r"\(.*?\)","",txt).strip()
        dishes=sorted({_strip(x)
                       for c in ("Breakfast","Lunch","Dinner")
                       for x in S["plan"][c]})
        dsel=st.selectbox("Dish",dishes)
        if st.button("Generate recipe"):
            if not OPENAI_AVAILABLE:
                st.error("OpenAI key missing")
            else:
                try:
                    sys="You are a Ghanaian recipe encyclopedia."
                    prompt=(f"Recipe for **{dsel}** in ≈{int(dailyk//3)} kcal. "
                            "Use Ghanaian household measures & clear steps.")
                    rsp=client_openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"system","content":sys},
                                  {"role":"user","content":prompt}],
                        temperature=.7,timeout=30)
                    txt=rsp.choices[0].message.content.strip()
                    st.markdown(txt)
                    st.download_button("Save recipe.txt",txt.encode(),"recipe.txt")
                except Exception as e:
                    st.error(e)

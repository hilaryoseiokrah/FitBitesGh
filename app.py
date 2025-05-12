###############################################################################
# FitBites – full app (2025-05-13)
# • classic_plan now shows kcal (not grams)
# • registration inline-if bug fixed
# • profile has explicit Save button
# • partial & full reshuffle protected against KeyError/NULL
###############################################################################

import os, glob, json, datetime, re
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"          # makes Streamlit auto-reload

import streamlit as st
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────── Optional OpenAI (GPT) ──────────────────────
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    client_openai = OpenAI(api_key=_KEY) if _KEY else None
    OPENAI_AVAILABLE = bool(client_openai)
except Exception:
    client_openai = None

def _rerun(): st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

# ──────────────────────── Paths & helpers ─────────────────────────
DATA_DIR, PROFILE_DIR, PIC_DIR, MEALPLAN_DIR = (
    "data", "data/profiles", "data/profile_pics", "data/mealplans"
)
for p in (DATA_DIR, PROFILE_DIR, PIC_DIR, MEALPLAN_DIR): os.makedirs(p, exist_ok=True)

USER_FILE = f"{DATA_DIR}/users.csv"
def users_df(): return pd.read_csv(USER_FILE) if os.path.isfile(USER_FILE) else pd.DataFrame(columns=["username","password"])
def save_user(u, p):
    df = users_df()
    if u in df.username.values: return False
    df = pd.concat([df, pd.DataFrame([[u, p]], columns=df.columns)]).reset_index(drop=True)
    df.to_csv(USER_FILE, index=False); return True
def valid(u,p): return not users_df()[(users_df().username==u)&(users_df().password==p)].empty

def prof_path(u): return f"{PROFILE_DIR}/{u}.json"
def pic_path(u):  return f"{PIC_DIR}/{u}.png"
def csv_path(u,w):return f"{MEALPLAN_DIR}/{u}_week{w}.csv"
def weeks(u):     return [int(f.split("week")[-1].split(".")[0]) for f in glob.glob(f"{MEALPLAN_DIR}/{u}_week*.csv")]

def load_prof(u):
    if os.path.isfile(prof_path(u)): return json.load(open(prof_path(u)))
    return dict(weight=90,height=160,age=25,sex="female",activity="sedentary",
                target_weight=75,likes_b=[],likes_l=[],likes_d=[],
                dislikes=[],use_ai=False,last_updated=str(datetime.date.today()))
def save_prof(u,d): json.dump(d,open(prof_path(u),"w"),indent=2)

# ─────────────────────── Streamlit session ────────────────────────
defaults = dict(logged_in=False, username="", profile={}, meal_plan=None,
                current_week=None, daily_calories=None, show_reshuffle=False)
for k,v in defaults.items(): st.session_state.setdefault(k,v)

st.set_page_config("FitBites – AI Ghanaian Meal Plans", layout="wide")

# ╔═════════════════════ LOGIN / REGISTER ════════════════════╗
if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")
    tab_login, tab_reg = st.tabs(["Login", "Register"])

    with tab_login:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if valid(u, p):
                st.session_state.update(logged_in=True, username=u, profile=load_prof(u)); _rerun()
            else: st.error("Invalid credentials")

    with tab_reg:
        nu  = st.text_input("Choose Username")
        npw = st.text_input("Choose Password", type="password")
        if st.button("Create account"):
            if save_user(nu, npw): st.success("Account created!")
            else: st.warning("Username already exists")

    st.stop()   # ── end early until logged-in

# ╚════════════════════════════════════════════════════════════╝

# ───── sidebar logout ─────
with st.sidebar:
    if st.button("🚪 Log out"):
        st.session_state.clear(); st.session_state.logged_in=False; _rerun()

# ───────────────────────── Welcome banner ─────────────────────────
st.markdown(f"""
#### 👋 Wo ho te sɛn, **{st.session_state.username.title()}**?  
> *Adidie pa yɛ ahoɔden pa* — **Good food equals good health** 🇬🇭
""")

# ───────────────────────── Load food + embeddings ────────────────
@st.cache_data
def load_food():
    df = pd.read_csv("gh_food_nutritional_values.csv")
    df.Food = df.Food.str.strip().str.lower()
    cols = ["Protein(g)","Fat(g)","Carbs(g)","Calories(100g)","Water(g)","SFA(100g)","MUFA(100g)","PUFA(100g)"]
    df[cols] = df[cols].fillna(df[cols].mean())
    return df, cols
df, ncols = load_food()

class AE(nn.Module):
    def __init__(s,d):
        super().__init__()
        s.e = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,8))
        s.d = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,d))
    def forward(s,x): z=s.e(x); return z, s.d(z)

@st.cache_resource
def embed(mat):
    net = AE(mat.shape[1]); opt = optim.Adam(net.parameters(), 1e-3)
    loss = nn.MSELoss(); t = torch.tensor(mat, dtype=torch.float32)
    for _ in range(200):
        opt.zero_grad(); z,out = net(t); loss(out,t).backward(); opt.step()
    with torch.no_grad(): z,_ = net(t); return z.numpy()
emb = embed(StandardScaler().fit_transform(df[ncols]))

def similar(food, k=5, exc=None):
    exc = exc or []
    idx = df.index[df.Food == food][0]
    sims = cosine_similarity(emb[idx].reshape(1,-1), emb).ravel()
    return [df.iloc[i].Food for i in sims.argsort()[::-1] if df.iloc[i].Food not in exc and df.iloc[i].Food != food][:k]

# ────────────────────── TDEE & classic plan ───────────────────────
def tdee(w,h,age,sex,act):
    mult = dict(sedentary=1.2, light=1.375, moderate=1.55, active=1.725, superactive=1.9)[act]
    return (10*w + 6.25*h - 5*age + (5 if sex=="male" else -161)) * mult

def classic_plan(prefs, kcal, dislikes):
    split = dict(breakfast=0.25, lunch=0.35, dinner=0.4)
    out   = []
    for d in range(1,8):
        row, tot = {"Day":f"Day {d}"}, 0
        for meal, f in split.items():
            opts=[]
            for s in prefs.get(meal, []): opts.extend(similar(s, exc=dislikes))
            if not opts: opts = list(set(df.Food.unique()) - set(dislikes))
            if not opts:
                pick, meal_kcal = "No food", 0
            else:
                pick, meal_kcal = np.random.choice(opts), kcal*f
            row[meal.capitalize()] = f"{pick} ({meal_kcal:.0f} kcal)"; tot += meal_kcal
        row["Total Calories"] = f"{tot:.0f} kcal"; out.append(row)
    return pd.DataFrame(out)

# ───────────────────── GPT plan (optional) ────────────────────────
def gpt_plan(prefs, dislikes, kcal):
    if not OPENAI_AVAILABLE: return None
    likes = ", ".join(set(sum(prefs.values(), []))) or "any Ghanaian foods"
    dis   = ", ".join(dislikes) if dislikes else "none"
    rules = ("Pairings:\n"
             "• Hausa koko must pair with bofrot or bread & groundnuts\n"
             "• Fufu pairs with any soup except okro stew\n"
             "• Banku pairs with okro soup + protein\n"
             "• No fictional Ghanaian foods\n")
    sys  = "You are a Ghanaian dietitian. Reply ONLY with minified JSON list."
    user = (f"{rules}\nBuild 7-day table (Day,Breakfast,Lunch,Dinner). "
            f"Daily ≈{int(kcal)} kcal (25/35/40). Use household measures & kcal per item. "
            f"LIKES:{likes}. DISLIKES:{dis}.")
    try:
        r   = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},
                          {"role":"user"  ,"content":user}],
                temperature=0.4, timeout=30)
        raw = r.choices[0].message.content.strip()
        j   = raw[raw.find("["):raw.rfind("]")+1]
        return pd.DataFrame(json.loads(j))
    except Exception as e:
        st.error(f"GPT error: {e}"); return None

# ═══════════════ Tabs ═══════════════
tab_plan, tab_profile, tab_recipe = st.tabs(["🍽️ Planner","👤 Profile","🍲 Recipe"])

# ─────────────── Planner tab ───────────────
with tab_plan:
    prof = st.session_state.profile
    # Sidebar inputs
    with st.sidebar:
        st.subheader("Details")
        w  = st.number_input("Current weight (kg)", 30.0,200.0,float(prof["weight"]))
        tw = st.number_input("Target weight (kg)", 30.0,200.0,float(prof["target_weight"]))
        h  = st.number_input("Height (cm)",        120.0,250.0,float(prof["height"]))
        age= st.number_input("Age",10,100,int(prof["age"]))
        sex= st.selectbox("Sex",["female","male"],index=0 if prof["sex"]=="female" else 1)
        act= st.selectbox("Activity",["sedentary","light","moderate","active","superactive"],
                          index=["sedentary","light","moderate","active","superactive"].index(prof["activity"]))
        st.subheader("Likes")
        likes_b = st.multiselect("Breakfast", df.Food.unique(), default=prof["likes_b"])
        likes_l = st.multiselect("Lunch",     df.Food.unique(), default=prof["likes_l"])
        likes_d = st.multiselect("Dinner",    df.Food.unique(), default=prof["likes_d"])
        st.subheader("Dislikes")
        dislikes= st.multiselect("Dislikes",  df.Food.unique(), default=prof["dislikes"])
        use_ai  = st.checkbox("🤖 Use AI combos", value=prof.get("use_ai",False), disabled=not OPENAI_AVAILABLE)

    def generate_plan(next_week=False):
        kcal   = tdee(w,h,age,sex,act) - 500
        prefs  = dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
        week   = (max(weeks(st.session_state.username) or [0])+1) if next_week else (st.session_state.current_week or 1)
        plan   = gpt_plan(prefs, dislikes, kcal) if use_ai else classic_plan(prefs, kcal, dislikes)
        if plan is not None:
            plan.to_csv(csv_path(st.session_state.username, week), index=False)
            st.session_state.meal_plan, st.session_state.current_week = plan, week
            st.session_state.daily_calories = kcal
            months = abs(w - tw) / 2
            st.success(f"✅ Week {week} saved. Daily kcal: {int(kcal)}")
            st.info   (f"⏳ Estimated time to reach {tw} kg: **{months:.1f} months**")
            # save profile
            prof.update(weight=w,height=h,age=age,sex=sex,activity=act,target_weight=tw,
                        likes_b=likes_b,likes_l=likes_l,likes_d=likes_d,
                        dislikes=dislikes,use_ai=use_ai,last_updated=str(datetime.date.today()))
            save_prof(st.session_state.username, prof)
            st.session_state.profile = prof

    col1,col2 = st.columns(2)
    if col1.button("✨ Generate / Update"): generate_plan(False)
    if col2.button("➕ Next week"):          generate_plan(True)

    # load existing week selection
    if weeks(st.session_state.username):
        sel = st.selectbox("Week", weeks(st.session_state.username),
                           index=weeks(st.session_state.username).index(st.session_state.current_week or weeks(st.session_state.username)[-1]))
        if sel != st.session_state.current_week:
            st.session_state.current_week = sel
            st.session_state.meal_plan = pd.read_csv(csv_path(st.session_state.username, sel))

    # show plan + reshuffle
    if st.session_state.meal_plan is not None:
        st.dataframe(st.session_state.meal_plan, use_container_width=True)
        st.download_button("⬇️ CSV", st.session_state.meal_plan.to_csv(index=False).encode(),
                           file_name=f"mealplan_week{st.session_state.current_week}.csv")
        if st.button("🔄 Reshuffle"): st.session_state.show_reshuffle=True

    # Reshuffle helpers
    def ensure_kcal(): return st.session_state.daily_calories or (tdee(w,h,age,sex,act)-500)

    if st.session_state.show_reshuffle and st.session_state.meal_plan is not None:
        st.markdown("### 🔄 Reshuffle Plan")
        mode = st.radio("Type",["Partial","Full"], horizontal=True)

        if mode=="Partial":
            days  = st.multiselect("Days", st.session_state.meal_plan.Day.tolist())
            meals = st.multiselect("Meals",["Breakfast","Lunch","Dinner"])
            extra = st.multiselect("Extra dislikes", df.Food.unique())
            if st.button("Apply partial"):
                prefs   = dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
                upd_dis = list(set(dislikes + extra))
                new     = gpt_plan(prefs, upd_dis, ensure_kcal()) if use_ai else classic_plan(prefs, ensure_kcal(), upd_dis)

                if new is not None:
                    missing = [d for d in days if d not in new.Day.values]
                    if missing: st.warning("Days not in reshuffle: " + ", ".join(missing))
                    for d in days:
                        if d not in new.Day.values: continue
                        oi = st.session_state.meal_plan[st.session_state.meal_plan.Day == d].index[0]
                        ni = new[new.Day == d].index[0]
                        for m in meals:
                            if m in new.columns and m in st.session_state.meal_plan.columns:
                                st.session_state.meal_plan.at[oi, m] = new.at[ni, m]
                        if "Total Calories" in new.columns:
                            st.session_state.meal_plan.at[oi,"Total Calories"] = new.at[ni,"Total Calories"]
                    st.session_state.meal_plan.to_csv(csv_path(st.session_state.username, st.session_state.current_week), index=False)
                    st.session_state.show_reshuffle=False; _rerun()

        else:  # Full reshuffle
            extra = st.multiselect("Extra dislikes", df.Food.unique(), key="full_dis")
            if st.button("Apply full"):
                prefs   = dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
                upd_dis = list(set(dislikes + extra))
                new     = gpt_plan(prefs, upd_dis, ensure_kcal()) if use_ai else classic_plan(prefs, ensure_kcal(), upd_dis)
                if new is not None:
                    st.session_state.meal_plan = new
                    new.to_csv(csv_path(st.session_state.username, st.session_state.current_week), index=False)
                    st.session_state.show_reshuffle=False; _rerun()

# ─────────────── Profile tab ───────────────
with tab_profile:
    st.header("👤 Profile")
    p = st.session_state.profile
    c1,c2 = st.columns([1,3])
    with c1:
        if os.path.isfile(pic_path(st.session_state.username)):
            st.image(pic_path(st.session_state.username), width=180)
        up = st.file_uploader("Upload photo", type=["png","jpg","jpeg"])
        if up:
            open(pic_path(st.session_state.username),"wb").write(up.getbuffer())
            st.success("Photo saved")
    with c2:
        st.markdown(f"""
* **Weight:** {p['weight']} kg  
* **Height:** {p['height']} cm  
* **Target weight:** {p['target_weight']} kg  
* **Age:** {p['age']}  
* **Sex:** {p['sex'].title()}  
* **Activity:** {p['activity'].title()}  
* **Last update:** {p['last_updated']}
""")
        if st.button("💾 Save Profile"):
            save_prof(st.session_state.username, p); st.success("Profile saved")

    st.divider(); st.subheader("📦 Saved Plans")
    for w in weeks(st.session_state.username):
        with st.expander(f"Week {w}"):
            st.download_button("Download CSV", open(csv_path(st.session_state.username,w),"rb").read(),
                               file_name=f"mealplan_week{w}.csv", key=f"dl{w}")

# ─────────────── Recipe tab ───────────────
def recipe_llm(dish, kcal):
    if not OPENAI_AVAILABLE: return None
    sys  = "You are a Ghanaian recipe dictionary."
    user = f"Create a recipe for {dish} ≈{int(kcal)} kcal using household measures and clear steps."
    try:
        r = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":sys},
                          {"role":"user","content":user}],
                temperature=0.7, timeout=30)
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT error: {e}"); return None

with tab_recipe:
    st.header("🍲 Recipe Maker")
    if st.session_state.meal_plan is None:
        st.info("Generate a meal plan first.")
    else:
        clean = lambda t: re.sub(r"\s*\\(.*?\\)", "", t).strip()
        dishes = sorted({clean(x) for col in ["Breakfast","Lunch","Dinner"] for x in st.session_state.meal_plan[col]})
        dish   = st.selectbox("Choose a dish", dishes)
        if st.button("Generate recipe"):
            kcal = ensure_kcal() // 3
            with st.spinner("Generating recipe…"):
                rec = recipe_llm(dish, kcal)
            if rec:
                st.markdown(rec)
                st.download_button("Save txt", rec.encode(), "recipe.txt", "text/plain")

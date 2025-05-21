###############################################################################
#  FitBites – Ghana-centric AI Meal-Planner
#  Rev. 2025-05-21  •  full feature set: BMI/TDEE, ANN recs, GPT meal combos,
#                    partial/full reshuffle, profile, next-week plans & recipes
###############################################################################

import os, glob, json, datetime as dt, re, io, random
from pathlib import Path

# ─────────────────────────────── Third-party ────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────── OpenAI (optional) ──────────────────────────────
try:
    from openai import OpenAI                # pip install --upgrade openai
    client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY",
                                             st.secrets.get("OPENAI_API_KEY","")))
    OPENAI_AVAILABLE = True
except Exception:
    client_openai, OPENAI_AVAILABLE = None, False

# ──────────────────────────────── Paths ─────────────────────────────────────
DATA     = Path("data")
PROFILES = DATA / "profiles"
PICS     = DATA / "profile_pics"
PLANS    = DATA / "mealplans"
for p in (DATA, PROFILES, PICS, PLANS): p.mkdir(parents=True, exist_ok=True)

USERS_CSV = DATA / "users.csv"

# ───────────────────────────── Auth helpers ─────────────────────────────────
def _users_df() -> pd.DataFrame:
    if USERS_CSV.exists():
        return pd.read_csv(USERS_CSV)
    return pd.DataFrame(columns=["username", "password"])

def register_user(username:str, password:str) -> bool:
    df = _users_df()
    if username in df.username.values:
        return False
    df.loc[len(df)] = [username, password]
    df.to_csv(USERS_CSV, index=False)
    return True

def authenticate(username:str, password:str) -> bool:
    df = _users_df()
    return not df[(df.username==username)&(df.password==password)].empty

# ─────────────────────────── Profile persistence ────────────────────────────
def profile_path(u:str) -> Path:      return PROFILES / f"{u}.json"
def picture_path(u:str) -> Path:      return PICS / f"{u}.png"
def plan_path(u:str, w:int) -> Path:  return PLANS / f"{u}_week{w}.csv"

def load_profile(u:str) -> dict:
    if profile_path(u).exists():
        return json.loads(profile_path(u).read_text())
    # defaults
    return dict(
        weight=90, height=160, age=25, sex="female",
        activity="sedentary", target_weight=75,
        likes_b=[], likes_l=[], likes_d=[], dislikes=[],
        use_ai=False, last_updated=str(dt.date.today())
    )

def save_profile(u:str, obj:dict) -> None:
    profile_path(u).write_text(json.dumps(obj, indent=2))

def existing_weeks(u:str) -> list[int]:
    weeks = [re.search(r"week(\d+)", f.name) for f in PLANS.glob(f"{u}_week*.csv")]
    return sorted(int(m.group(1)) for m in weeks if m)

# ─────────────────────── Calculation helpers (BMI, TDEE) ───────────────────
def bmi(weight_kg:float, height_cm:float) -> float:
    return weight_kg / (height_cm/100)**2

def tdee_mifflin_stjeor(weight:float, height:float, age:int, sex:str,
                        activity:str) -> float:
    base = 10*weight + 6.25*height - 5*age + (5 if sex=="male" else -161)
    factors = dict(sedentary=1.2, light=1.375, moderate=1.55,
                   active=1.725, superactive=1.9)
    return base * factors[activity]

SAFE_RATE_KG_PER_WEEK = 0.75        # change here if you want a different rate

def weeks_to_goal(current_w:float, target_w:float) -> float:
    delta = abs(current_w - target_w)
    return delta / SAFE_RATE_KG_PER_WEEK

# ──────────────────────── Load food + build embeddings ─────────────────────
@st.cache_data(show_spinner=False)
def load_food() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv("gh_food_nutritional_values.csv")
    df["Food"] = df["Food"].str.strip().str.lower()
    nutrient_cols = ["Protein(g)", "Fat(g)", "Carbs(g)", "Calories(100g)",
                     "Water(g)", "SFA(100g)", "MUFA(100g)", "PUFA(100g)"]
    df[nutrient_cols] = df[nutrient_cols].fillna(df[nutrient_cols].mean())
    return df, nutrient_cols
FOODS_DF, NUTR_COLS = load_food()

class AutoEncoder(nn.Module):
    def __init__(self, in_dim:int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(),
                                     nn.Linear(32, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(),
                                     nn.Linear(32, in_dim))
    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

@st.cache_resource(show_spinner=False)
def build_embeddings(mat:np.ndarray) -> np.ndarray:
    torch.manual_seed(7)
    model = AutoEncoder(mat.shape[1])
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    tmat  = torch.tensor(mat, dtype=torch.float32)

    for _ in range(300):
        opt.zero_grad()
        z, recon = model(tmat)
        lossf(recon, tmat).backward()
        opt.step()

    with torch.no_grad():
        z, _ = model(tmat)
        return z.numpy()

STD_SCALER = StandardScaler()
EMB        = build_embeddings(STD_SCALER.fit_transform(FOODS_DF[NUTR_COLS]))

def top_similar(food:str, k:int=5, exclude:list[str]|None=None) -> list[str]:
    exclude = exclude or []
    idx     = FOODS_DF.index[FOODS_DF.Food==food]
    if idx.empty:                                   # food not found
        return []
    idx = idx[0]
    sims = cosine_similarity(EMB[idx].reshape(1,-1), EMB).ravel()
    ordered = sims.argsort()[::-1]
    out = []
    for i in ordered:
        name = FOODS_DF.iloc[i].Food
        if name != food and name not in exclude:
            out.append(name)
        if len(out) == k:
            break
    return out

# ───────────────────────────────── Streamlit UI ─────────────────────────────
st.set_page_config("FitBites – AI Ghanaian Meal Plans", layout="wide")

# ── Session defaults
if "state" not in st.session_state:
    st.session_state.state = dict(
        logged_in=False, username="", profile={},
        meal_plan=None, current_week=None,
        daily_kcal=None
    )

S = st.session_state.state   # alias for brevity

# ───────────────────────────── Login / Register ─────────────────────────────
if not S["logged_in"]:
    st.title("🔐 FitBites Login")
    tab_login, tab_reg = st.tabs(("Login", "Register"))

    with tab_login:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(u, p):
                S.update(dict(logged_in=True, username=u, profile=load_profile(u)))
                st.experimental_rerun()
            else:
                st.error("Incorrect username/password")

    with tab_reg:
        nu  = st.text_input("Choose username")
        npw = st.text_input("Choose password", type="password")
    
        if st.button("Create account"):
            ok = register_user(nu, npw)          # ← this line actually writes CSV
            if ok:
                st.success("Account created")    # user + pass now saved
            else:
                st.warning("User exists")

    st.stop()

# ──────────────────────────── Main Sidebar – inputs ─────────────────────────
with st.sidebar:
    st.subheader(f"Hi, {S['username'].title()}")

    # quick-access vars for convenience
    P = S["profile"]
    weight = st.number_input("Current weight (kg)", 30.0, 200.0, float(P["weight"]), 0.1)
    target = st.number_input("Target weight (kg)", 30.0, 200.0, float(P["target_weight"]), 0.1)
    height = st.number_input("Height (cm)", 120.0, 250.0, float(P["height"]), 0.1)
    age    = st.number_input("Age", 10, 100, int(P["age"]), 1)
    sex    = st.selectbox("Sex", ["female", "male"],
                          index=0 if P["sex"]=="female" else 1)
    activity = st.selectbox("Activity level",
                            ["sedentary", "light", "moderate", "active", "superactive"],
                            index=["sedentary","light","moderate","active","superactive"].index(P["activity"]))

    st.markdown("---")
    st.subheader("Meal preferences")

    likes_b = st.multiselect("Breakfast likes", FOODS_DF.Food.unique(), default=P["likes_b"])
    likes_l = st.multiselect("Lunch likes"    , FOODS_DF.Food.unique(), default=P["likes_l"])
    likes_d = st.multiselect("Dinner likes"   , FOODS_DF.Food.unique(), default=P["likes_d"])
    dislikes= st.multiselect("Dislikes", FOODS_DF.Food.unique(), default=P["dislikes"])

    use_ai  = st.checkbox("Use GPT to build meal combos", value=P["use_ai"],
                          disabled=not OPENAI_AVAILABLE)

    if st.button("Save profile"):
        P.update(dict(weight=weight, height=height, age=age, sex=sex,
                      activity=activity, target_weight=target,
                      likes_b=likes_b, likes_l=likes_l, likes_d=likes_d,
                      dislikes=dislikes, use_ai=use_ai,
                      last_updated=str(dt.date.today())))
        save_profile(S["username"], P)
        st.success("Profile saved")

    if st.button("🚪 Log out"):
        st.session_state.clear()
        st.experimental_rerun()

# ───────────────── TV-style banner with live metrics ────────────────────────
col1, col2, col3 = st.columns(3)
bmi_now = bmi(weight, height)
tdee    = tdee_mifflin_stjeor(weight, height, age, sex, activity)
daily_k = tdee - 500                         # 500 kcal deficit – tweak if needed
S["daily_kcal"] = daily_k
weeks_goal = weeks_to_goal(weight, target)

col1.metric("BMI", f"{bmi_now:.1f}")
col2.metric("TDEE", f"{int(tdee)} kcal/day")
col3.metric("Time to goal",
            f"{int(weeks_goal)} weeks" if weeks_goal>1 else "≈1 week")

st.divider()

# ────────────────────────── Meal-plan generation helpers ───────────────────
def classic_plan(prefs:dict, kcal:float, dislikes:list[str]) -> pd.DataFrame:
    """
    Very simple heuristic: pick one food for each meal type,
    scale grams so that kcal ≈ split * daily_target.
    """
    split = dict(breakfast=.25, lunch=.35, dinner=.40)
    out = []
    for day in range(1, 8):
        row, grams_total = {"Day": f"Day {day}"}, 0
        for meal, frac in split.items():
            pool = []
            for liked in prefs.get(meal, []):
                pool.extend(top_similar(liked, k=5, exclude=dislikes))
            if not pool:
                # fall back to random Ghanaian foods not in dislikes
                pool = [f for f in FOODS_DF.Food.sample(30) if f not in dislikes]
            choice = random.choice(pool)
            cals100 = FOODS_DF.loc[FOODS_DF.Food==choice, "Calories(100g)"].iat[0]
            grams   = kcal*frac / cals100 * 100
            row[meal.capitalize()] = f"{choice} ({grams:.0f} g)"
            grams_total += grams
        row["Total (g)"] = f"{grams_total:.0f}"
        out.append(row)
    return pd.DataFrame(out)

def gpt_meal_plan(prefs:dict, dislikes:list[str], kcal:float) -> pd.DataFrame|None:
    if not OPENAI_AVAILABLE: return None
    likes_str = ", ".join(set(sum(prefs.values(), []))) or "any Ghanaian foods"
    dis_str   = ", ".join(dislikes) if dislikes else "none"

    rules = (
        "Pairings:\n"
        "• Hausa koko or koko must go with bofrot or bread & groundnuts\n"
        "• Fufu pairs with any soup except okro (no stews)\n"
        "• Greek yoghurt goes with banana\n"
        "• Yam or cassava pair with stew\n"
        "• Rice pairs with stews\n"
        "• Banku / akple pair with okro soup + protein\n"
        "• No imaginary dishes – stick to Ghanaian database foods\n"
    )

    sys_msg = "You are a registered Ghanaian dietitian. Output JSON only."
    user_msg = (
       f"{rules}\n\n"
       f"Create a 7-day table (keys = Day, Breakfast, Lunch, Dinner). "
       f"Fit each day into ≈{int(kcal)} kcal (25/35/40 split). "
       f"LIKES: {likes_str}.  DISLIKES: {dis_str}. "
       f"Use household measures & include kcal per item."
    )

    try:
        resp = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys_msg},
                      {"role":"user","content":user_msg}],
            temperature=.3, timeout=30)
        raw = resp.choices[0].message.content
        json_block = raw[raw.find("["): raw.rfind("]")+1]
        return pd.DataFrame(json.loads(json_block))
    except Exception as e:
        st.error(f"GPT meal-plan error: {e}")
        return None

def generate_plan(next_week:bool=False):
    prefs = dict(breakfast=likes_b, lunch=likes_l, dinner=likes_d)
    week  = (max(existing_weeks(S["username"]) or [0]) + 1) if next_week \
            else (S["current_week"] or 1)

    if use_ai:
        plan_df = gpt_meal_plan(prefs, dislikes, daily_k)
    else:
        plan_df = classic_plan(prefs, daily_k, dislikes)

    if plan_df is None:
        return

    plan_df.to_csv(plan_path(S["username"], week), index=False)
    S.update(dict(meal_plan=plan_df, current_week=week))

    st.success(f"Week {week} plan saved")

# ───────────────────────────── Tabs layout ──────────────────────────────────
tab_plan, tab_profile, tab_recipe = st.tabs(("🍽 Planner", "👤 Profile", "🍲 Recipe"))

# ::::::::::::::::::::::::::::::::  PLANNER  :::::::::::::::::::::::::::::::::
with tab_plan:
    colA, colB = st.columns(2)
    if colA.button("Generate / update this week"):
        generate_plan(next_week=False)
    if colB.button("Generate next-week plan"):
        generate_plan(next_week=True)

    # Week selector if any plans exist
    weeks_avail = existing_weeks(S["username"])
    if weeks_avail:
        wk = st.selectbox("Select week",
                          weeks_avail,
                          index = weeks_avail.index(S["current_week"]
                                                    or weeks_avail[-1]))
        if wk != S.get("current_week"):
            S["current_week"] = wk
            S["meal_plan"] = pd.read_csv(plan_path(S["username"], wk))

    # Show current plan
    if S["meal_plan"] is not None:
        st.dataframe(S["meal_plan"], use_container_width=True)
        st.download_button("Download CSV",
                           S["meal_plan"].to_csv(index=False).encode(),
                           f"mealplan_week{S['current_week']}.csv")
        st.markdown("---")

        # ── Reshuffle controls
        with st.expander("Reshuffle meals"):
            mode = st.radio("Mode", ("Partial", "Full"), horizontal=True)
            if mode == "Partial":
                sel_days  = st.multiselect("Days", S["meal_plan"].Day.tolist())
                sel_meals = st.multiselect("Meals", ["Breakfast","Lunch","Dinner"])
                extra_dis = st.multiselect("Extra temporary dislikes", FOODS_DF.Food.unique())
                if st.button("Apply partial reshuffle"):
                    prefs_now = dict(breakfast=likes_b, lunch=likes_l, dinner=likes_d)
                    combined_dislikes = list(set(dislikes + extra_dis))
                    new_plan = (gpt_meal_plan if use_ai else classic_plan)(
                        prefs_now, combined_dislikes, daily_k
                    )
                    if new_plan is not None:
                        # Replace only selected cells
                        for day in sel_days:
                            i_old = S["meal_plan"].index[S["meal_plan"].Day == day][0]
                            i_new = new_plan.index[new_plan.Day == day][0]
                            for meal in sel_meals:
                                S["meal_plan"].at[i_old, meal] = new_plan.at[i_new, meal]
                            # keep Total(g) in sync
                            S["meal_plan"].at[i_old, "Total (g)"] = new_plan.at[i_new, "Total (g)"]
                        S["meal_plan"].to_csv(plan_path(S["username"], S["current_week"]),
                                              index=False)
                        st.success("Partial reshuffle applied")
            else:   # Full
                extra_dis = st.multiselect("Extra temporary dislikes", FOODS_DF.Food.unique())
                if st.button("Apply full reshuffle"):
                    prefs_now = dict(breakfast=likes_b, lunch=likes_l, dinner=likes_d)
                    combined_dislikes = list(set(dislikes + extra_dis))
                    new_plan = (gpt_meal_plan if use_ai else classic_plan)(
                        prefs_now, combined_dislikes, daily_k
                    )
                    if new_plan is not None:
                        S["meal_plan"] = new_plan
                        new_plan.to_csv(plan_path(S["username"], S["current_week"]), index=False)
                        st.success("Full reshuffle applied")

# ::::::::::::::::::::::::::::::::  PROFILE  :::::::::::::::::::::::::::::::::
with tab_profile:
    st.header("Your profile")
    pc1, pc2 = st.columns([1,3])
    with pc1:
        if picture_path(S["username"]).exists():
            st.image(picture_path(S["username"]), width=180)
        upl = st.file_uploader("Upload profile photo", type=["png","jpg","jpeg"])
        if upl:
            picture_path(S["username"]).write_bytes(upl.getbuffer())
            st.success("Photo saved – refresh if not visible")
    with pc2:
        p = S["profile"]
        st.markdown(
            f"""
            * **Weight:** {p['weight']} kg  
            * **Height:** {p['height']} cm  
            * **Target weight:** {p['target_weight']} kg  
            * **Age:** {p['age']}  
            * **Sex:** {p['sex'].title()}  
            * **Activity:** {p['activity'].title()}  
            * **Last update:** {p['last_updated']}
            """
        )

    st.divider()
    st.subheader("Saved meal plans")
    for w in existing_weeks(S["username"]):
        with st.expander(f"Week {w}"):
            st.download_button(
                label="Download CSV",
                data=open(plan_path(S["username"], w),"rb").read(),
                file_name=f"mealplan_week{w}.csv",
                key=f"dl{w}"
            )

# ::::::::::::::::::::::::::::::::: RECIPE :::::::::::::::::::::::::::::::::::
with tab_recipe:
    st.header("Recipe generator")
    if S["meal_plan"] is None:
        st.info("Generate a meal plan first")
    else:
        # pull unique dishes (strip trailing "(xx g)" if present)
        def _clean(txt:str) -> str:
            return re.sub(r"\s*\(.*?\)\s*$","",txt).strip()
        dishes = sorted({_clean(x)
                         for col in ("Breakfast","Lunch","Dinner")
                         for x in S["meal_plan"][col]})
        dish = st.selectbox("Pick a dish", dishes)
        if st.button("Generate recipe"):
            if not OPENAI_AVAILABLE:
                st.error("OpenAI key not configured")
            else:
                daily_portion = daily_k // 3
                with st.spinner("Calling GPT…"):
                    try:
                        sys = "You are a Ghanaian recipe encyclopedia."
                        prompt = (f"Give me a recipe for **{dish}** that fits about "
                                  f"{int(daily_portion)} kcal. "
                                  "Use Ghanaian household measures. Steps + kcal.")
                        rsp = client_openai.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role":"system","content":sys},
                                      {"role":"user","content":prompt}],
                            temperature=.7, timeout=30)
                        recipe = rsp.choices[0].message.content.strip()
                        st.markdown(recipe)
                        st.download_button("Save recipe.txt",
                                           recipe.encode(),
                                           "recipe.txt", "text/plain")
                    except Exception as e:
                        st.error(f"GPT recipe error: {e}")

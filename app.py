###############################################################################
# FitBites – full app (2025-05-12)
###############################################################################
# • Login / Register (CSV)
# • Profile tab: info + profile picture + plan history
# • Weekly meal planner: AI GPT-4o combos or classic auto-encoder
# • Reshuffles, CSV download
# • AI Recipe Maker
###############################################################################

import os, glob, io, json, datetime
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"  # avoid torch watcher crash

import streamlit as st
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────── OpenAI (optional) ─────────────────────────
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    client_openai = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
    OPENAI_AVAILABLE = bool(client_openai)
except Exception:
    client_openai = None

def _rerun(): st.rerun() if hasattr(st, "rerun") else st.experimental_rerun()

# ───────────────────────── Paths & folders ───────────────────────────
DATA_DIR, PROFILE_DIR, PIC_DIR, MEALPLAN_DIR = "data", "data/profiles", "data/profile_pics", "data/mealplans"
for p in (DATA_DIR, PROFILE_DIR, PIC_DIR, MEALPLAN_DIR):
    os.makedirs(p, exist_ok=True)

# ───────────────────────── Auth helpers ──────────────────────────────
USER_FILE = f"{DATA_DIR}/users.csv"

def users_df():
    return pd.read_csv(USER_FILE) if os.path.isfile(USER_FILE) else pd.DataFrame(columns=["username", "password"])

def save_user(u, p):
    df = users_df()
    if u in df.username.values:
        return False
    pd.concat([df, pd.DataFrame([[u, p]], columns=df.columns)]).to_csv(USER_FILE, index=False)
    return True

def valid_user(u, p):
    df = users_df()
    return not df[(df.username == u) & (df.password == p)].empty

# ───────────────────────── Profile helpers ───────────────────────────
def prof_path(u):     return f"{PROFILE_DIR}/{u}.json"
def pic_path(u):      return f"{PIC_DIR}/{u}.png"
def meal_csv(u, w):   return f"{MEALPLAN_DIR}/{u}_week{w}.csv"
def week_list(u):     return [int(f.split("week")[-1].split(".")[0]) for f in glob.glob(f"{MEALPLAN_DIR}/{u}_week*.csv")]

def load_profile(u):
    if os.path.isfile(prof_path(u)):
        return json.load(open(prof_path(u)))
    return dict(weight=90, height=160, age=25, sex="female", activity="sedentary",
                target_weight=75, likes_b=[], likes_l=[], likes_d=[],
                dislikes=[], use_ai=False, last_updated=str(datetime.date.today()))

def save_profile(u, data):
    json.dump(data, open(prof_path(u), "w"), indent=2)

# ───────────────────────── Session defaults ──────────────────────────
defaults = dict(
    logged_in=False, username="", profile={},
    meal_plan=None,  current_week=None, daily_calories=None,
)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

st.set_page_config("FitBites – AI Ghanaian Meal Plans", layout="wide")

# ───────────────────────── LOGIN / REGISTER ──────────────────────────
if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        user = st.text_input("Username")
        pwd  = st.text_input("Password", type="password")
        if st.button("Login"):
            if valid_user(user, pwd):
                st.session_state.logged_in = True
                st.session_state.username  = user
                st.session_state.profile   = load_profile(user)
                _rerun()
            else:
                st.error("Invalid credentials")

    with tab_register:
        new_u = st.text_input("Choose Username")
        new_p = st.text_input("Choose Password", type="password")
        if st.button("Create account"):
            if save_user(new_u, new_p):
                st.success("Account created! Switch to Login tab.")
            else:
                st.warning("Username already exists")
    st.stop()

# ───────────────────────── Logout button ─────────────────────────────
with st.sidebar:
    if st.button("🚪 Log out"):
        st.session_state.clear()
        st.session_state.logged_in = False
        _rerun()

# ───────────────────────── Load food + embeddings ────────────────────
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
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 8))
        self.d = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, d))
    def forward(self, x):
        z = self.e(x)
        return z, self.d(z)

@st.cache_resource
def get_embeddings(mat):
    net, opt, loss = AE(mat.shape[1]), optim.Adam(AE(mat.shape[1]).parameters(), 1e-3), nn.MSELoss()
    t = torch.tensor(mat, dtype=torch.float32)
    for _ in range(200):
        opt.zero_grad()
        z, out = net(t)
        loss(out, t).backward()
        opt.step()
    with torch.no_grad():
        z, _ = net(t)
        return z.numpy()
emb = get_embeddings(StandardScaler().fit_transform(df[nut_cols]))

def similar(food, k=5, exclude=None):
    exclude = exclude or []
    idx = df.index[df.Food == food][0]
    sims = cosine_similarity(emb[idx].reshape(1, -1), emb).ravel()
    return [df.iloc[i].Food for i in sims.argsort()[::-1]
            if df.iloc[i].Food not in exclude and df.iloc[i].Food != food][:k]

# ───────────────────────── Planner helpers ───────────────────────────
def tdee(w, h, a, sex, act):
    mult = dict(sedentary=1.2, light=1.375, moderate=1.55, active=1.725, superactive=1.9)[act]
    return (10*w + 6.25*h - 5*a + (5 if sex == "male" else -161)) * mult

def classic_plan(prefs, kcal, dislikes):
    split = dict(breakfast=0.25, lunch=0.35, dinner=0.4)
    rows = []
    for d in range(1, 8):
        row, tot = {"Day": f"Day {d}"}, 0
        for meal, frac in split.items():
            opts = []
            for s in prefs.get(meal, []):
                opts.extend(similar(s, exclude=dislikes))
            if not opts:
                opts = list(set(df.Food.sample(5)) - set(dislikes))
            pick = np.random.choice(opts)
            cal100 = df.loc[df.Food == pick, "Calories(100g)"].iat[0]
            grams  = kcal * frac / cal100 * 100
            row[meal.capitalize()] = f"{pick} ({grams:.0f}g)"
            tot += grams
        row["Total Portion (g)"] = f"{tot:.0f}g"
        rows.append(row)
    return pd.DataFrame(rows)

def gpt_plan(prefs, dislikes, kcal):
    if not OPENAI_AVAILABLE:
        return None
    likes = ", ".join(set(sum(prefs.values(), []))) or "any Ghanaian foods"
    dis   = ", ".join(dislikes) if dislikes else "none"
    prompt = (
        f"You are a Ghanaian dietitian. Build a 7-day JSON table (Day, Breakfast, "
        f"Lunch, Dinner) using Ghanaian household measures. Daily ≈{int(kcal)} kcal "
        f"(25/35/40). LIKES: {likes}. DISLIKES: {dis}. Include kcal in parentheses."
    )
    try:
        r = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=30,
        )
        return pd.read_json(io.StringIO(r.choices[0].message.content.strip()))
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None

def recipe_llm(ing, cui):
    if not OPENAI_AVAILABLE:
        return None
    sys = "You are a recipe dictionary. Only respond with recipes based on the user's inputs..."
    user = f"Ingredients: {ing}. Cuisine: {cui}."
    try:
        r = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.7,
            timeout=30,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None

# ───────────────────────── Tabs ──────────────────────────
tab_plan, tab_profile, tab_recipe = st.tabs(["🍽️ Planner", "👤 Profile", "🍲 Recipe Maker"])

# ====================  TAB 1: Planner  ====================
with tab_plan:
    st.header("Weekly Meal Planner")

    prof = st.session_state.profile  # shorthand

    with st.sidebar:
        st.subheader("Details")
        w   = st.number_input("Weight (kg)", 30.0, 200.0, float(prof["weight"]), step=0.1)
        h   = st.number_input("Height (cm)", 120.0, 250.0, float(prof["height"]), step=0.1)
        age = st.number_input("Age", 10, 100, int(prof["age"]), step=1)
        sex = st.selectbox("Sex", ["female", "male"], index=0 if prof["sex"]=="female" else 1)
        act = st.selectbox("Activity", ["sedentary","light","moderate","active","superactive"],
                           index=["sedentary","light","moderate","active","superactive"].index(prof["activity"]))

        st.subheader("Likes")
        likes_b = st.multiselect("Breakfast", df.Food.unique(), default=prof.get("likes_b", []))
        likes_l = st.multiselect("Lunch",     df.Food.unique(), default=prof.get("likes_l", []))
        likes_d = st.multiselect("Dinner",    df.Food.unique(), default=prof.get("likes_d", []))

        st.subheader("Dislikes")
        dislikes = st.multiselect("Dislikes", df.Food.unique(), default=prof.get("dislikes", []))

        use_ai = st.checkbox("🤖 Use AI combos", value=prof.get("use_ai", False),
                             disabled=not OPENAI_AVAILABLE)

        def save_plan(next_week: bool):
            st.session_state.daily_calories = tdee(w, h, age, sex, act) - 500
            prefs = dict(breakfast=likes_b, lunch=likes_l, dinner=likes_d)

            if next_week:
                week = max(week_list(st.session_state.username) or [0]) + 1
            else:
                week = st.session_state.current_week or 1

            plan = (
                gpt_plan(prefs, dislikes, st.session_state.daily_calories)
                if use_ai else
                classic_plan(prefs, st.session_state.daily_calories, dislikes)
            )
            if plan is not None:
                plan.to_csv(meal_csv(st.session_state.username, week), index=False)
                st.session_state.meal_plan = plan
                st.session_state.current_week = week
                st.success(f"Week {week} saved")

            # update & persist profile
            prof.update(
                weight=w, height=h, age=int(age), sex=sex, activity=act,
                likes_b=likes_b, likes_l=likes_l, likes_d=likes_d,
                dislikes=dislikes, use_ai=use_ai,
                last_updated=str(datetime.date.today()),
            )
            save_profile(st.session_state.username, prof)
            st.session_state.profile = prof

        if st.button("✨ Generate / Update"):
            save_plan(next_week=False)
        if st.button("➕ Next week"):
            save_plan(next_week=True)

    # Week picker & table
    wks = week_list(st.session_state.username)
    if wks:
        selected = st.selectbox("Week", wks,
                                index=wks.index(st.session_state.current_week or wks[-1]))
        if selected != st.session_state.current_week:
            st.session_state.current_week = selected
            st.session_state.meal_plan   = pd.read_csv(meal_csv(st.session_state.username, selected))

    if st.session_state.meal_plan is not None:
        st.dataframe(st.session_state.meal_plan, use_container_width=True)
        st.download_button("⬇️ Download CSV",
                           st.session_state.meal_plan.to_csv(index=False).encode(),
                           file_name=f"mealplan_week{st.session_state.current_week}.csv")

# ====================  TAB 2: Profile  ====================
with tab_profile:
    st.header("Profile")
    p = st.session_state.profile

    col1, col2 = st.columns([1, 3])
    with col1:
        if os.path.isfile(pic_path(st.session_state.username)):
            st.image(pic_path(st.session_state.username), width=180)
        uploaded = st.file_uploader("Upload / replace photo", type=["png", "jpg", "jpeg"])
        if uploaded:
            open(pic_path(st.session_state.username), "wb").write(uploaded.getbuffer())
            st.success("Saved. Refresh.")
    with col2:
        st.markdown(f"""
* **Weight:** {p['weight']} kg  
* **Height:** {p['height']} cm  
* **Target weight:** {p['target_weight']} kg  
* **Age:** {p['age']}  
* **Sex:** {p['sex'].title()}  
* **Activity:** {p['activity'].title()}  
* **Last updated:** {p['last_updated']}
""")
    st.divider()
    st.subheader("Saved plans")
    for w in week_list(st.session_state.username):
        with st.expander(f"Week {w}"):
            st.download_button("Download CSV",
                               open(meal_csv(st.session_state.username, w), "rb").read(),
                               file_name=f"mealplan_week{w}.csv",
                               key=f"d{w}")

# ====================  TAB 3: Recipe Maker  ====================
with tab_recipe:
    st.header("AI Recipe Maker")
    ing = st.text_area("Ingredients (comma-separated)")
    cui = st.text_input("Cuisine (e.g. Ghanaian, Italian)")
    if st.button("Generate recipe"):
        if not ing.strip() or not cui.strip():
            st.warning("Please fill both fields.")
        else:
            with st.spinner("GPT cooking..."):
                recipe = recipe_llm(ing, cui)
            if recipe:
                st.markdown(recipe)
                st.download_button("⬇️ Save txt", recipe.encode(),
                                   file_name="recipe.txt", mime="text/plain")


# ===== Profile Tab =====
with tab_profile:
    st.header("Profile")
    p = st.session_state.profile
    col1, col2 = st.columns([1, 3])
    with col1:
        if os.path.isfile(pic_path(st.session_state.username)):
            st.image(pic_path(st.session_state.username), width=150)
        up = st.file_uploader("Upload profile picture", type=["png", "jpg", "jpeg"])
        if up:
            open(pic_path(st.session_state.username), "wb").write(up.getbuffer())
            st.success("Saved. Refresh.")
    with col2:
        st.markdown(f"""
* **Weight:** {p['weight']} kg  
* **Height:** {p['height']} cm  
* **Target weight:** {p['target_weight']} kg  
* **Age:** {p['age']}  
* **Sex:** {p['sex'].title()}  
* **Activity:** {p['activity'].title()}  
* **Last updated:** {p['last_updated']}
""")
    st.divider()
    st.subheader("Saved plans")
    for w in weeks(st.session_state.username):
        with st.expander(f"Week {w}"):
            st.download_button("Download", open(ppath(st.session_state.username, w), "rb").read(),
                               file_name=f"mealplan_week{w}.csv", key=f"d{w}")

# ===== Recipe Tab =====
with tab_recipe:
    st.header("AI Recipe Maker")
    ing = st.text_area("Ingredients (comma-separated)")
    cui = st.text_input("Cuisine")
    if st.button("Generate recipe"):
        if not ing.strip() or not cui.strip():
            st.warning("Please fill both fields.")
        else:
            def recipe_llm(ing, cui):
                if not OPENAI_AVAILABLE: return None
                sys = "You are a recipe dictionary. Only respond with recipes."
                user = f"Ingredients: {ing}. Cuisine: {cui}."
                try:
                    r = client_openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                        temperature=0.7,
                        timeout=30,
                    )
                    return r.choices[0].message.content.strip()
                except Exception as e:
                    st.error(f"OpenAI error: {e}")
                    return None

            with st.spinner("GPT cooking…"):
                r = recipe_llm(ing, cui)
            if r:
                st.markdown(r)
                st.download_button("⬇️ Save txt", r.encode(), file_name="recipe.txt", mime="text/plain")

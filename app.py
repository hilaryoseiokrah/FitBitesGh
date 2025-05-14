###############################################################################
# FitBites – Weekly Meal-Plan Edition  +  AI Ghanaian Combos  +  Recipe Maker
# (save as app.py, put gh_food_nutritional_values.csv in same folder)
###############################################################################

import os, glob, io, json, time
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"   # avoid PyTorch watcher crash

import streamlit as st
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --------------- optional OpenAI ----------------
OPENAI_AVAILABLE = False
try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    OPENAI_AVAILABLE = bool(openai.api_key)
except Exception:
    pass

# ---------- safe rerun for all Streamlit versions ----------
def _safe_rerun():
    if hasattr(st, "rerun"): st.rerun()
    else:                    st.experimental_rerun()

st.set_page_config(page_title="FitBites – AI Ghanaian Meal Plans 🇬🇭",
                   layout="wide")

# ────────────────────────── CSV auth utils ────────────────────────────────
USER_FILE = "users.csv"
def load_users():
    return pd.read_csv(USER_FILE) if os.path.exists(USER_FILE) else pd.DataFrame(columns=["username","password"])
def save_user(u,p):
    users = load_users()
    if u in users.username.values: return False
    pd.concat([users, pd.DataFrame([[u,p]], columns=["username","password"])]
             ).to_csv(USER_FILE,index=False); return True
def valid_login(u,p): return not load_users()[(load_users().username==u)&(load_users().password==p)].empty

# ────────────────────────── session defaults ──────────────────────────────
defaults = dict(
    logged_in=False, username="",
    meal_plan=None,   daily_calories=None,
    reshuffle_mode=False, current_week=None,
)
for k,v in defaults.items(): st.session_state.setdefault(k,v)

# ────────────────────────── LOGIN / REGISTER ──────────────────────────────
if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")
    tab_login, tab_reg = st.tabs(["Login","Register"])
    with tab_login:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if valid_login(u,p):
                st.session_state.logged_in=True
                st.session_state.username=u
                _safe_rerun()
            else: st.error("❌ Invalid credentials")

    with tab_reg:
        nu  = st.text_input("Choose Username", key="reg_user")
        npw = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register", key="reg_btn"):
            st.success("✅ Registered! Switch to Login.") if save_user(nu,npw) else st.warning("Username exists")
    st.stop()

# ────────────────────────── LOGOUT ────────────────────────────────────────
with st.sidebar:
    if st.button("🚪 Log Out"):
        for k in list(st.session_state.keys()):
            if k!="logged_in": st.session_state.pop(k,None)
        st.session_state.logged_in=False
        _safe_rerun()

# ────────────────────────── BANNER & GUIDE ────────────────────────────────
st.markdown(f"""
### 👋 Wo ho te sɛn, **{st.session_state.username.title()}**?

> *Adidie pa yɛ ahoɔden pa* — **Good food equals good health** 🇬🇭

**How to use FitBites**  
1. Fill in your details on the left.  
2. *(Optional)* choose foods you **like** and **avoid**.  
3. Check **AI-generated combos** to let GPT-4 craft full Ghanaian meals.  
4. Click **✨ Generate Plan** or **➕ Next Week**.  
5. Download CSV or reshuffle. Use the **Recipe Maker** to turn ingredients \
   into a full recipe.

---
""", unsafe_allow_html=True)

# ────────────────────────── FOOD DATA + EMBEDDINGS ────────────────────────
@st.cache_data
def load_food():
    df = pd.read_csv("gh_food_nutritional_values.csv")
    df.Food = df.Food.str.strip().str.lower()
    cols = ["Protein(g)","Fat(g)","Carbs(g)","Calories(100g)",
            "Water(g)","SFA(100g)","MUFA(100g)","PUFA(100g)"]
    df[cols]=df[cols].fillna(df[cols].mean()); return df, cols
df, nut_cols = load_food()

class AE(nn.Module):
    def __init__(self,d): super().__init__(); self.e=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,8)); self.d=nn.Sequential(nn.Linear(8,32),nn.ReLU(),nn.Linear(32,d))
    def forward(self,x): z=self.e(x); return z, self.d(z)
@st.cache_resource
def embed(mat):
    net=AE(mat.shape[1]); opt=optim.Adam(net.parameters(),1e-3); loss=nn.MSELoss(); t=torch.tensor(mat,dtype=torch.float32)
    for _ in range(200): opt.zero_grad(); z,out=net(t); loss(out,t).backward(); opt.step()
    with torch.no_grad(): z,_=net(t); return z.numpy()
emb = embed(StandardScaler().fit_transform(df[nut_cols]))
def similar(food,k=5,exclude=None):
    exclude=exclude or []; idx=df.index[df.Food==food][0]
    sims=cosine_similarity(emb[idx].reshape(1,-1),emb).ravel()
    return [df.iloc[i].Food for i in sims.argsort()[::-1] if df.iloc[i].Food not in exclude and df.iloc[i].Food!=food][:k]

# ────────────────────────── classic plan builder ──────────────────────────
def calc_tdee(w,h,a,sex,act):
    bmr=10*w+6.25*h-5*a+(5 if sex=="male" else -161)
    return bmr*dict(sedentary=1.2,light=1.375,moderate=1.55,active=1.725,superactive=1.9)[act]
def build_plan(prefs,kcal,dislikes):
    split=dict(breakfast=0.25,lunch=0.35,dinner=0.4); rows=[]
    for d in range(1,8):
        row={"Day":f"Day {d}"}; tot=0
        for meal,f in split.items():
            opts=[]; [opts.extend(similar(s,exclude=dislikes)) for s in prefs.get(meal,[])]
            if not opts: opts=list(set(df.Food.sample(5))-set(dislikes))
            pick=np.random.choice(opts); cal100=df.loc[df.Food==pick,"Calories(100g)"].iat[0]
            grams=kcal*f/cal100*100; row[meal.capitalize()]=f"{pick} ({grams:.0f}g)"; tot+=grams
        row["Total Portion (g)"]=f"{tot:.0f}g"; rows.append(row)
    return pd.DataFrame(rows)

# ─────────────── OpenAI meal-plan generator (new API) ───────────────
def gpt_meal_plan(prefs, dislikes, daily_kcal):
    if not OPENAI_AVAILABLE:
        st.error("OpenAI key missing."); return None

    from openai import OpenAI                               # new import
    client = OpenAI(api_key=openai.api_key)                 # re-use key

    likes = ", ".join(set(sum(prefs.values(), []))) or "any Ghanaian foods"
    dis   = ", ".join(dislikes) if dislikes else "none"
    prompt = f"""
You are a Ghanaian dietitian. Build a 7-day table of balanced meals using household
measures (scoops, ladles, cups, pieces). Daily goal ≈ {int(daily_kcal)} kcal:
Breakfast 25 %, Lunch 35 %, Dinner 40 %.
LIKES: {likes}
DISLIKES: {dis}
Return ONLY valid JSON list of 7 objects with keys Day, Breakfast, Lunch, Dinner.
Each meal string must show portions & kcal in parentheses.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=30,
        )
        return pd.read_json(io.StringIO(resp.choices[0].message.content.strip()))
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None


# ─────────────── OpenAI recipe generator (new API) ───────────────
def generate_recipe_llm(ingredients, cuisine):
    if not OPENAI_AVAILABLE:
        st.error("OpenAI key missing."); return None

    from openai import OpenAI
    client = OpenAI(api_key=openai.api_key)

    prompt = (
        f"You are a recipe dictionary. Here is a list of ingredients: {ingredients}. "
        f"The cuisine is {cuisine}. Use only the available ingredients to provide a recipe for a meal."
    )
    sys_msg = (
        "You are a recipe dictionary. Only respond with recipes based on the user's "
        "inputs. Add specific quantities during the recipe instructions and rate the "
        "recipe at the beginning on a scale of 1-5. Throw an error message when "
        "anything other than ingredients is inputed. Don't modify the ingredients; "
        "use them as they are."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            timeout=30,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None


# file helpers
def plan_path(user,week): return f"mealplans_{user}_week{week}.csv"
def list_weeks(user):
    return [int(f.split('week')[-1].split('.csv')[0]) for f in sorted(glob.glob(f"mealplans_{user}_week*.csv"))]

# ────────────────────────── SIDEBAR INPUTS ────────────────────────────────
with st.sidebar:
    st.subheader("📋 Your Details")
    w=st.number_input("Weight (kg)",30,200,90); h=st.number_input("Height (cm)",120,250,160)
    age=st.number_input("Age",10,100,25); sex=st.selectbox("Sex",["female","male"])
    act=st.selectbox("Activity",["sedentary","light","moderate","active","superactive"])

    st.subheader("🍽 Likes (optional)")
    likes_b=st.multiselect("Breakfast",df.Food.unique())
    likes_l=st.multiselect("Lunch",df.Food.unique())
    likes_d=st.multiselect("Dinner",df.Food.unique())

    st.subheader("🚫 Dislikes")
    dislikes=st.multiselect("Never include",df.Food.unique())

    use_ai=st.checkbox("🤖 Use AI-generated combos", value=False, disabled=not OPENAI_AVAILABLE)

    if st.button("✨ Generate Plan"):
        st.session_state.daily_calories=calc_tdee(w,h,age,sex,act)-500
        prefs=dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
        week=st.session_state.current_week or (max(list_weeks(st.session_state.username))+1 if list_weeks(st.session_state.username) else 1)
        plan = gpt_meal_plan(prefs,dislikes,st.session_state.daily_calories) if use_ai else build_plan(prefs,st.session_state.daily_calories,dislikes)
        if plan is not None:
            plan.to_csv(plan_path(st.session_state.username,week),index=False)
            st.session_state.meal_plan=plan; st.session_state.current_week=week; st.session_state.reshuffle_mode=False
            st.success(f"Week {week} saved")

    if st.button("➕ Generate Next Week Plan"):
        st.session_state.daily_calories=calc_tdee(w,h,age,sex,act)-500
        prefs=dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)
        next_week=(max(list_weeks(st.session_state.username))+1) if list_weeks(st.session_state.username) else 1
        plan = gpt_meal_plan(prefs,dislikes,st.session_state.daily_calories) if use_ai else build_plan(prefs,st.session_state.daily_calories,dislikes)
        if plan is not None:
            plan.to_csv(plan_path(st.session_state.username,next_week),index=False)
            st.session_state.meal_plan=plan; st.session_state.current_week=next_week; st.session_state.reshuffle_mode=False
            st.success(f"Week {next_week} saved")

# ────────────────────────── WEEK PICKER ───────────────────────────────────
weeks=list_weeks(st.session_state.username)
if weeks:
    sel_week=st.selectbox("📆 View week",weeks,index=weeks.index(st.session_state.current_week or weeks[-1]))
    if sel_week!=st.session_state.current_week:
        st.session_state.current_week=sel_week
        st.session_state.meal_plan=pd.read_csv(plan_path(st.session_state.username,sel_week))
        st.session_state.reshuffle_mode=False

# ────────────────────────── PLAN DISPLAY ──────────────────────────────────
if st.session_state.meal_plan is not None:
    st.subheader(f"📅 Week {st.session_state.current_week} Meal Plan")
    st.dataframe(st.session_state.meal_plan,use_container_width=True)
    st.download_button("⬇️ Download this plan",
                       st.session_state.meal_plan.to_csv(index=False).encode(),
                       file_name=f"mealplan_week{st.session_state.current_week}.csv",
                       mime="text/csv")
    if st.button("🔄 Reshuffle Plan"): st.session_state.reshuffle_mode=True

# ────────────────────────── RESHUFFLE UI / LOGIC ──────────────────────────
if st.session_state.reshuffle_mode and st.session_state.meal_plan is not None:
    st.markdown("---"); st.markdown("### 🔄 Reshuffle Options")
    mode=st.radio("Choose",["Partial","Full"],horizontal=True)
    prefs=dict(breakfast=likes_b,lunch=likes_l,dinner=likes_d)

    if mode=="Partial":
        days_sel=st.multiselect("Days",st.session_state.meal_plan.Day.tolist())
        meals_sel=st.multiselect("Meals",["Breakfast","Lunch","Dinner"])
        extra_dis=st.multiselect("Extra dislikes",df.Food.unique())
        if st.button("Apply Partial"):
            upd_dis=list(set(dislikes+extra_dis))
            new=gpt_meal_plan(prefs,upd_dis,st.session_state.daily_calories) if (use_ai and OPENAI_AVAILABLE) else build_plan(prefs,st.session_state.daily_calories,upd_dis)
            if new is not None:
                for d in days_sel:
                    old_i=st.session_state.meal_plan.index[st.session_state.meal_plan.Day==d][0]
                    new_i=new.index[new.Day==d][0]
                    for m in meals_sel: st.session_state.meal_plan.at[old_i,m]=new.at[new_i,m]
                    st.session_state.meal_plan.at[old_i,"Total Portion (g)"]=new.at[new_i,"Total Portion (g)"]
                st.session_state.meal_plan.to_csv(plan_path(st.session_state.username,st.session_state.current_week),index=False)
                st.session_state.reshuffle_mode=False; _safe_rerun()

    if mode=="Full":
        extra_full=st.multiselect("Extra dislikes for NEW plan",df.Food.unique())
        if st.button("Apply Full Reshuffle"):
            upd_dis=list(set(dislikes+extra_full))
            st.session_state.meal_plan=gpt_meal_plan(prefs,upd_dis,st.session_state.daily_calories) if (use_ai and OPENAI_AVAILABLE) else build_plan(prefs,st.session_state.daily_calories,upd_dis)
            if st.session_state.meal_plan is not None:
                st.session_state.meal_plan.to_csv(plan_path(st.session_state.username,st.session_state.current_week),index=False)
                st.session_state.reshuffle_mode=False; _safe_rerun()

# ────────────────────────── AI RECIPE MAKER (NEW) ─────────────────────────
st.markdown("---"); st.markdown("## 🍲 AI Recipe Maker")
with st.expander("Generate a recipe from your own ingredients"):
    ing = st.text_area("Ingredients (comma-separated)")
    cui = st.text_input("Cuisine (e.g. Ghanaian, Italian)")
    if st.button("Generate recipe 🎉"):
        if not ing.strip() or not cui.strip():
            st.warning("Please enter both ingredients and cuisine.")
        else:
            with st.spinner("Crafting your recipe …"):
                recipe = generate_recipe_llm(ing, cui)
            if recipe:
                st.markdown("### Your recipe")
                st.markdown(recipe)
                st.download_button("⬇️ Download recipe",
                                   data=recipe.encode(),
                                   file_name="recipe.txt",
                                   mime="text/plain")
###############################################################################
# FitBites App – Weekly Meal-Plan Edition
###############################################################################
#  • Login / register (CSV)
#  • Autoencoder food similarity
#  • 7-day meal plan with likes/dislikes, partial/full reshuffle
#  • NEW: week history + “next week” generation + CSV download
#  • Twi proverb + updated how-to guide
###############################################################################

import os, glob, io
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"

import streamlit as st
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ────────────────────────── compatibility rerun ──────────────────────────────
def _safe_rerun():
    if hasattr(st, "rerun"):  st.rerun()
    else:                     st.experimental_rerun()
# ────────────────────────── page config ──────────────────────────────────────
st.set_page_config(page_title="FitBites – Personalized Meal Plans 🇬🇭",
                   layout="wide")
# ────────────────────────── auth helpers ─────────────────────────────────────
USER_FILE = "users.csv"
def load_users():
    return pd.read_csv(USER_FILE) if os.path.exists(USER_FILE) else pd.DataFrame(columns=["username","password"])
def save_user(u,p):
    users = load_users()
    if u in users.username.values: return False
    pd.concat([users,pd.DataFrame([[u,p]],columns=["username","password"])]
             ).to_csv(USER_FILE,index=False); return True
def valid_login(u,p): return not load_users()[(load_users().username==u)&(load_users().password==p)].empty
# ────────────────────────── session state defaults ───────────────────────────
for k,v in dict(logged_in=False, username="", meal_plan=None,
                reshuffle_mode=False, daily_calories=None,
                current_week=None).items():
    st.session_state.setdefault(k,v)
# ────────────────────────── LOGIN / REGISTER UI ──────────────────────────────
if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")
    login_tab, reg_tab = st.tabs(["Login","Register"])
    with login_tab:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if valid_login(u,p):
                st.session_state.logged_in=True; st.session_state.username=u; _safe_rerun()
            else: st.error("❌ Invalid credentials")
    with reg_tab:
        nu = st.text_input("Choose Username", key="reg_user")
        npw= st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register", key="reg_btn"):
            st.success("✅ Registered! Switch to Login.") if save_user(nu,npw) else st.warning("Username exists")
    st.stop()

# ────────────────────────── LOG OUT ──────────────────────────────────────────
with st.sidebar:
    if st.button("🚪 Log Out"):
        for k in list(st.session_state.keys()):
            if k!="logged_in": st.session_state.pop(k,None)
        st.session_state.logged_in=False; _safe_rerun()

# ────────────────────────── WELCOME BANNER & GUIDE ───────────────────────────
st.markdown(f"""
### 👋 Wo ho te sɛn, **{st.session_state.username.title()}**?

🎉 **Welcome to FitBites!**

> *Adidie pa yɛ ahoɔden pa* — **Good food equals good health** 🇬🇭

#### 📝 How to use FitBites
1. Fill in your details in the sidebar.  
2. *(Optional)* choose foods you **like** and foods to **avoid**.  
3. Click **✨ Generate Plan** or **Generate Next Week Plan** for a new week.  
4. Use **🔄 Reshuffle Plan** to tweak specific meals or the whole week.  
5. Click **⬇️ Download this plan** to save the current week’s CSV.  

---
""", unsafe_allow_html=True)

# ────────────────────────── load food + embeddings (cached) ───────────────────
@st.cache_data
def load_food():
    df = pd.read_csv("gh_food_nutritional_values.csv")
    df.Food = df.Food.str.strip().str.lower()
    cols=["Protein(g)","Fat(g)","Carbs(g)","Calories(100g)","Water(g)","SFA(100g)","MUFA(100g)","PUFA(100g)"]
    df[cols]=df[cols].fillna(df[cols].mean()); return df,cols
df, nut_cols = load_food()

class AE(nn.Module):
    def __init__(self,d): super().__init__(); self.enc=nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Linear(32,16))
    def forward(self,x): z=self.enc(x); return z
@st.cache_resource
def embed(mat):
    net=AE(mat.shape[1]); opt=optim.Adam(net.parameters(),1e-3); loss=nn.MSELoss(); t=torch.tensor(mat,dtype=torch.float32)
    for _ in range(200):
        opt.zero_grad(); z=net(t); loss(z,t[:,:16]).backward(); opt.step()
    with torch.no_grad(): return net(t).numpy()
emb = embed(StandardScaler().fit_transform(df[nut_cols]))

def similar(food,k=5,exclude=None):
    exclude=exclude or []; idx=df.index[df.Food==food][0]
    sims=cosine_similarity(emb[idx].reshape(1,-1),emb).ravel()
    return [df.iloc[i].Food for i in sims.argsort()[::-1] if df.iloc[i].Food not in exclude and df.iloc[i].Food!=food][:k]

# ────────────────────────── plan utilities ───────────────────────────────────
def calc_tdee(w,h,a,sex,act):
    bmr=10*w+6.25*h-5*a+(5 if sex=="male" else -161)
    return bmr*dict(sedentary=1.2,light=1.375,moderate=1.55,active=1.725,superactive=1.9)[act]
def build_plan(prefs,kcal,dislikes):
    split={"breakfast":0.25,"lunch":0.35,"dinner":0.4}; rows=[]
    for d in range(1,8):
        row={"Day":f"Day {d}"}; tot=0
        for meal,f in split.items():
            opts=[]; [opts.extend(similar(s,exclude=dislikes)) for s in prefs.get(meal,[])]
            if not opts: opts=list(set(df.Food.sample(5))-set(dislikes))
            pick=np.random.choice(opts); cal100=df.loc[df.Food==pick,"Calories(100g)"].iat[0]
            g=kcal*f/cal100*100; row[meal.capitalize()]=f"{pick} ({g:.0f}g)"; tot+=g
        row["Total Portion (g)"]=f"{tot:.0f}g"; rows.append(row)
    return pd.DataFrame(rows)

def plan_path(user,week): return f"mealplans_{user}_week{week}.csv"
def list_weeks(user):
    files=sorted(glob.glob(f"mealplans_{user}_week*.csv"))
    return [int(f.split('week')[-1].split('.csv')[0]) for f in files]

# ────────────────────────── sidebar preferences ──────────────────────────────
with st.sidebar:
    st.subheader("📋 Details")
    w=st.number_input("Weight (kg)",30,200,90); h=st.number_input("Height (cm)",120,250,160)
    age=st.number_input("Age",10,100,25); sex=st.selectbox("Sex",["female","male"])
    act=st.selectbox("Activity",["sedentary","light","moderate","active","superactive"])
    st.subheader("🍽 Likes (optional)")
    likes_b=st.multiselect("Breakfast",df.Food.unique()); likes_l=st.multiselect("Lunch",df.Food.unique()); likes_d=st.multiselect("Dinner",df.Food.unique())
    st.subheader("🚫 Dislikes"); dislikes=st.multiselect("Never include",df.Food.unique())

    # ✓ Generate for first time or overwrite current week
    if st.button("✨ Generate Plan"):
        st.session_state.daily_calories=calc_tdee(w,h,age,sex,act)-500
        prefs={"breakfast":likes_b,"lunch":likes_l,"dinner":likes_d}
        plan=build_plan(prefs,st.session_state.daily_calories,dislikes)
        # decide week number
        weeks=list_weeks(st.session_state.username)
        week=st.session_state.current_week or (max(weeks) if weeks else 1)
        plan.to_csv(plan_path(st.session_state.username,week),index=False)
        st.session_state.meal_plan=plan; st.session_state.current_week=week; st.session_state.reshuffle_mode=False
        st.success(f"Week {week} generated & saved")

    # ✓ Generate next week (increment)
    if st.button("➕ Generate Next Week Plan"):
        st.session_state.daily_calories=calc_tdee(w,h,age,sex,act)-500
        prefs={"breakfast":likes_b,"lunch":likes_l,"dinner":likes_d}
        next_week=(max(list_weeks(st.session_state.username))+1) if list_weeks(st.session_state.username) else 1
        plan=build_plan(prefs,st.session_state.daily_calories,dislikes)
        plan.to_csv(plan_path(st.session_state.username,next_week),index=False)
        st.session_state.meal_plan=plan; st.session_state.current_week=next_week; st.session_state.reshuffle_mode=False
        st.success(f"Week {next_week} generated & saved")

# ────────────────────────── Week selector + load plan ─────────────────────────
available_weeks=list_weeks(st.session_state.username)
if available_weeks:
    selected_week=st.selectbox("📆 View week",available_weeks,index=available_weeks.index(st.session_state.current_week or available_weeks[-1]))
    if selected_week!=st.session_state.current_week:
        st.session_state.current_week=selected_week
        st.session_state.meal_plan=pd.read_csv(plan_path(st.session_state.username,selected_week))
        st.session_state.reshuffle_mode=False

# ────────────────────────── display plan & download ───────────────────────────
if st.session_state.meal_plan is not None:
    st.subheader(f"📅 Week {st.session_state.current_week} Meal Plan")
    st.dataframe(st.session_state.meal_plan,use_container_width=True)
    csv_bytes = st.session_state.meal_plan.to_csv(index=False).encode()
    st.download_button("⬇️ Download this plan", csv_bytes,
                       file_name=f"mealplan_week{st.session_state.current_week}.csv",
                       mime="text/csv")
    if st.button("🔄 Reshuffle Plan"): st.session_state.reshuffle_mode=True

# ────────────────────────── reshuffle UI/logic ───────────────────────────────
if st.session_state.reshuffle_mode and st.session_state.meal_plan is not None:
    st.markdown("---"); st.markdown("### 🔄 Reshuffle Options")
    mode=st.radio("Choose",["Partial","Full"],horizontal=True)
    prefs={"breakfast":likes_b,"lunch":likes_l,"dinner":likes_d}

    if mode=="Partial":
        days_sel=st.multiselect("Days",st.session_state.meal_plan.Day.tolist())
        meals_sel=st.multiselect("Meals",["Breakfast","Lunch","Dinner"])
        extra_dis=st.multiselect("Extra dislikes",df.Food.unique())
        if st.button("Apply Partial Reshuffle"):
            upd_dis=list(set(dislikes+extra_dis))
            new=build_plan(prefs,st.session_state.daily_calories,upd_dis)
            for day in days_sel:
                old_i=st.session_state.meal_plan.index[st.session_state.meal_plan.Day==day][0]
                new_i=new.index[new.Day==day][0]
                for m in meals_sel:
                    st.session_state.meal_plan.at[old_i,m]=new.at[new_i,m]
                st.session_state.meal_plan.at[old_i,"Total Portion (g)"]=new.at[new_i,"Total Portion (g)"]
            st.session_state.meal_plan.to_csv(plan_path(st.session_state.username,st.session_state.current_week),index=False)
            st.session_state.reshuffle_mode=False; _safe_rerun()

    if mode=="Full":
        extra_dis_full=st.multiselect("Extra dislikes for NEW plan",df.Food.unique())
        if st.button("Apply Full Reshuffle"):
            upd_dis=list(set(dislikes+extra_dis_full))
            st.session_state.meal_plan=build_plan(prefs,st.session_state.daily_calories,upd_dis)
            st.session_state.meal_plan.to_csv(plan_path(st.session_state.username,st.session_state.current_week),index=False)
            st.session_state.reshuffle_mode=False; _safe_rerun()

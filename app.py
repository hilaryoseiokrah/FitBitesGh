# -----------------------------
# FitBites App – FULL WORKING CODE
# -----------------------------
#   • CSV login / register
#   • Safe logout and page reruns on Streamlit ≥1.34
#   • Ghanaian-food autoencoder recommender
#   • 7-day calorie plan + partial / full reshuffle
#   • Plan saved to mealplans_<user>.csv
# -----------------------------

# --- Imports & env tweaks ---
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "poll"  # avoid torch watcher crash

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---- Compatible rerun helper -----------------
def _safe_rerun():
    """Works on both new (st.rerun) and old (st.experimental_rerun) versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()
# ---------------------------------------------

# MUST be first Streamlit command
st.set_page_config(page_title="FitBites – Personalized Meal Plans 🇬🇭", layout="wide")

# -----------------------------
# USER AUTH (CSV STORAGE)
# -----------------------------
USER_FILE = "users.csv"


def load_users():
    return (
        pd.read_csv(USER_FILE)
        if os.path.exists(USER_FILE)
        else pd.DataFrame(columns=["username", "password"])
    )


def save_user(username: str, password: str) -> bool:
    users = load_users()
    if username in users.username.values:
        return False
    users = pd.concat(
        [users, pd.DataFrame([[username, password]], columns=["username", "password"])]
    )
    users.to_csv(USER_FILE, index=False)
    return True


def valid_login(username: str, password: str) -> bool:
    users = load_users()
    return not users[
        (users.username == username) & (users.password == password)
    ].empty


# -----------------------------
# SESSION DEFAULTS
# -----------------------------
defaults = dict(
    logged_in=False,
    username="",
    meal_plan=None,
    reshuffle_mode=False,
    daily_calories=None,
)
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# -----------------------------
# LOGIN / REGISTER VIEW
# -----------------------------
if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")
    tab_login, tab_reg = st.tabs(["Login", "Register"])

    # --- Login tab ---
    with tab_login:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if valid_login(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                _safe_rerun()
            else:
                st.error("❌ Invalid credentials")

    # --- Register tab ---
    with tab_reg:
        nu = st.text_input("Choose Username", key="reg_user")
        npw = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register", key="reg_btn"):
            if save_user(nu, npw):
                st.success("✅ Registered! Switch to Login tab.")
            else:
                st.warning("Username already exists")

    st.stop()  # do not render the rest until logged in

# -----------------------------
# LOGOUT BUTTON
# -----------------------------
with st.sidebar:
    if st.button("🚪 Log Out"):
        for k in list(st.session_state.keys()):
            if k != "logged_in":
                st.session_state.pop(k, None)
        st.session_state.logged_in = False
        _safe_rerun()

# -----------------------------
# LOAD FOOD DATA + EMBEDDINGS
# -----------------------------
@st.cache_data
def load_food():
    df = pd.read_csv("gh_food_nutritional_values.csv")
    df.Food = df.Food.str.strip().str.lower()
    cols = [
        "Protein(g)",
        "Fat(g)",
        "Carbs(g)",
        "Calories(100g)",
        "Water(g)",
        "SFA(100g)",
        "MUFA(100g)",
        "PUFA(100g)",
    ]
    df[cols] = df[cols].fillna(df[cols].mean())
    return df, cols


df, nut_cols = load_food()


class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16))
        self.dec = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, d))

    def forward(self, x):
        z = self.enc(x)
        return z, self.dec(z)


@st.cache_resource
def get_embeddings(mat):
    net = AE(mat.shape[1])
    opt = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    t = torch.tensor(mat, dtype=torch.float32)
    for _ in range(200):
        opt.zero_grad()
        _, out = net(t)
        loss = loss_fn(out, t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net.enc(t).numpy()


X_scaled = StandardScaler().fit_transform(df[nut_cols])
emb = get_embeddings(X_scaled)

# -----------------------------
# UTILS
# -----------------------------
def similar(food, k=5, exclude=None):
    if exclude is None:
        exclude = []
    idx = df.index[df.Food == food][0]
    sims = cosine_similarity(emb[idx].reshape(1, -1), emb).ravel()
    order = sims.argsort()[::-1]
    return [
        df.iloc[i].Food
        for i in order
        if df.iloc[i].Food not in exclude and df.iloc[i].Food != food
    ][:k]


def calc_tdee(w, h, a, sex, act):
    bmr = 10 * w + 6.25 * h - 5 * a + (5 if sex == "male" else -161)
    mult = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "superactive": 1.9,
    }
    return bmr * mult[act]


def build_plan(prefs, kcal, dislikes):
    split = {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.4}
    rows = []
    for d in range(1, 8):
        row = {"Day": f"Day {d}"}
        total = 0
        for meal, frac in split.items():
            starters = prefs.get(meal, [])
            opts = []
            for s in starters:
                opts += similar(s, exclude=dislikes)
            if not opts:
                opts = list(set(df.Food.sample(5)) - set(dislikes))
            pick = np.random.choice(opts)
            cal100 = df.loc[df.Food == pick, "Calories(100g)"].iat[0]
            grams = kcal * frac / cal100 * 100
            row[meal.capitalize()] = f"{pick} ({grams:.0f}g)"
            total += grams
        row["Total Portion (g)"] = f"{total:.0f}g"
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# SIDEBAR FORM
# -----------------------------
with st.sidebar:
    st.subheader("📋 Details")
    w = st.number_input("Weight (kg)", 30, 200, 90)
    h = st.number_input("Height (cm)", 120, 250, 160)
    age = st.number_input("Age", 10, 100, 25)
    sex = st.selectbox("Sex", ["female", "male"])
    act = st.selectbox(
        "Activity", ["sedentary", "light", "moderate", "active", "superactive"]
    )

    st.subheader("🍽 Likes (optional)")
    likes_b = st.multiselect("Breakfast", df.Food.unique())
    likes_l = st.multiselect("Lunch", df.Food.unique())
    likes_d = st.multiselect("Dinner", df.Food.unique())

    st.subheader("🚫 Dislikes")
    dislikes = st.multiselect("Never include", df.Food.unique())

    if st.button("✨ Generate Plan"):
        st.session_state.daily_calories = calc_tdee(w, h, age, sex, act) - 500
        prefs = {"breakfast": likes_b, "lunch": likes_l, "dinner": likes_d}
        st.session_state.meal_plan = build_plan(
            prefs, st.session_state.daily_calories, dislikes
        )
        st.session_state.reshuffle_mode = False
        st.session_state.meal_plan.to_csv(
            f"mealplans_{st.session_state.username}.csv", index=False
        )
        st.success("Plan generated & saved")


# -----------------------------
# DISPLAY PLAN
# -----------------------------
if st.session_state.meal_plan is not None:
    st.subheader("📅 Your 7-Day Meal Plan")
    st.dataframe(st.session_state.meal_plan, use_container_width=True)
    if st.button("🔄 Reshuffle Plan"):
        st.session_state.reshuffle_mode = True

# -----------------------------
# RESHUFFLE LOGIC
# -----------------------------
if st.session_state.reshuffle_mode and st.session_state.meal_plan is not None:
    st.markdown("---")
    st.markdown("### 🔄 Reshuffle Options")
    mode = st.radio("Choose", ["Partial", "Full"], horizontal=True)

    # -------- Partial Reshuffle --------
    if mode == "Partial":
        days_sel = st.multiselect(
            "Select day(s) to reshuffle", st.session_state.meal_plan.Day.tolist()
        )
        meals_sel = st.multiselect("Which meals?", ["Breakfast", "Lunch", "Dinner"])
        extra_dis = st.multiselect(
            "Extra dislikes for this reshuffle", df.Food.unique()
        )

        if st.button("Apply Partial Reshuffle"):
            prefs = {"breakfast": likes_b, "lunch": likes_l, "dinner": likes_d}
            updated_dislikes = list(set(dislikes + extra_dis))
            new_plan = build_plan(
                prefs, st.session_state.daily_calories, updated_dislikes
            )

            for day in days_sel:
                old_idx = st.session_state.meal_plan.index[
                    st.session_state.meal_plan.Day == day
                ][0]
                new_idx = new_plan.index[new_plan.Day == day][0]
                for meal in meals_sel:
                    st.session_state.meal_plan.at[old_idx, meal] = new_plan.at[
                        new_idx, meal
                    ]
                st.session_state.meal_plan.at[
                    old_idx, "Total Portion (g)"
                ] = new_plan.at[new_idx, "Total Portion (g)"]

            st.session_state.meal_plan.to_csv(
                f"mealplans_{st.session_state.username}.csv", index=False
            )
            st.session_state.reshuffle_mode = False
            st.success("Partial reshuffle applied!")
            _safe_rerun()

    # -------- Full Reshuffle --------
    if mode == "Full":
        extra_dis_full = st.multiselect(
            "Extra dislikes for the NEW plan", df.Food.unique()
        )
        if st.button("Apply Full Reshuffle"):
            prefs = {"breakfast": likes_b, "lunch": likes_l, "dinner": likes_d}
            updated_dislikes = list(set(dislikes + extra_dis_full))
            st.session_state.meal_plan = build_plan(
                prefs, st.session_state.daily_calories, updated_dislikes
            )
            st.session_state.meal_plan.to_csv(
                f"mealplans_{st.session_state.username}.csv", index=False
            )
            st.session_state.reshuffle_mode = False
            st.success("Full reshuffle complete!")
            _safe_rerun()

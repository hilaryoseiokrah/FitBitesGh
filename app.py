# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- Auth: User Management ---
USER_FILE = "users.csv"

def load_users():
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    else:
        return pd.DataFrame(columns=["username", "password"])

def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_FILE, index=False)
    return True

def check_credentials(username, password):
    users = load_users()
    match = users[(users["username"] == username) & (users["password"] == password)]
    return not match.empty

# --- Streamlit Login ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "reshuffle_mode" not in st.session_state:
    st.session_state.reshuffle_mode = False
if "meal_plan" not in st.session_state:
    st.session_state.meal_plan = None

if not st.session_state.logged_in:
    st.title("🔐 Login to FitBites")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_credentials(login_user, login_pass):
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.success("✅ Logged in successfully! Please refresh the page manually.")
                st.stop()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")
        if st.button("Register"):
            if save_user(new_user, new_pass):
                st.success("🎉 Registered! You can now log in.")
            else:
                st.warning("Username already exists.")

    st.stop()

# --- Log Out ---
with st.sidebar:
    if st.button("🚪 Log Out"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.reshuffle_mode = False
        st.session_state.meal_plan = None
        st.experimental_rerun()

# --- Page Setup ---
st.set_page_config(page_title="FitBites - Personalized Meal Plans 🍽️", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('gh_food_nutritional_values.csv')
    df['Food'] = df['Food'].str.strip().str.lower()
    cols = ['Protein(g)', 'Fat(g)', 'Carbs(g)', 'Calories(100g)', 'Water(g)', 'SFA(100g)', 'MUFA(100g)', 'PUFA(100g)']
    df[cols] = df[cols].fillna(df[cols].mean())
    return df, cols

df, nutritional_columns = load_data()

# --- Neural Net ---
class FoodAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=16):
        super(FoodAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

@st.cache_resource
def train_model(X):
    model = FoodAutoencoder(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    for epoch in range(300):
        optimizer.zero_grad()
        embeddings, outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        food_embeddings = model.encoder(X_tensor).numpy()
    return food_embeddings

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[nutritional_columns])
food_embeddings = train_model(X_scaled)

# --- Helper Functions ---
def recommend_food_nn(food_name, dataset, embeddings, top_n=5, exclude=[]):
    dataset['Food'] = dataset['Food'].str.lower()
    food_name = food_name.lower()
    idx = dataset[dataset['Food'] == food_name].index[0]
    vec = embeddings[idx].reshape(1, -1)
    sims = cosine_similarity(vec, embeddings).flatten()
    recommended_idx = sims.argsort()[::-1]
    recommended_foods = []
    for i in recommended_idx:
        name = dataset.iloc[i]['Food']
        if name != food_name and name not in exclude:
            recommended_foods.append(name)
        if len(recommended_foods) >= top_n:
            break
    return recommended_foods

def calculate_bmi(weight, height_cm):
    return weight / (height_cm/100)**2

def calculate_tdee(weight, height, age, sex, activity_level):
    if sex == 'male':
        bmr = 10*weight + 6.25*height - 5*age + 5
    else:
        bmr = 10*weight + 6.25*height - 5*age - 161
    multipliers = {'sedentary':1.2, 'light':1.375, 'moderate':1.55, 'active':1.725, 'superactive':1.9}
    return bmr * multipliers[activity_level]

def generate_meal_plan(preferences, daily_calories, dislikes=[]):
    plan = []
    meal_split = {'breakfast':0.25, 'lunch':0.35, 'dinner':0.4}
    for day in range(1,8):
        day_plan = {'Day': f"Day {day}"}
        total_portion = 0
        for meal, portion in meal_split.items():
            starter_foods = preferences.get(meal, [])
            meal_foods = []
            for food in starter_foods:
                meal_foods += recommend_food_nn(food, df, food_embeddings, top_n=5, exclude=dislikes)
            meal_foods = list(set(meal_foods) - set(dislikes))
            if not meal_foods:
                meal_foods = list(set(df['Food'].sample(5).tolist()) - set(dislikes))
            selected = np.random.choice(meal_foods, 1)[0]
            cal = df[df['Food'] == selected]['Calories(100g)'].values[0]
            quantity = (daily_calories * portion) / cal * 100
            day_plan[meal.title()] = f"{selected} ({quantity:.0f}g)"
            total_portion += quantity
        day_plan['Total Portion (g)'] = f"{total_portion:.0f}g"
        plan.append(day_plan)
    return pd.DataFrame(plan)

# --- Inputs ---
st.title("🥗 FitBites Meal Planner")
st.sidebar.header("📋 Your Info")
weight = st.sidebar.number_input("Current Weight (kg)", 30, 200, 90)
target_weight = st.sidebar.number_input("Target Weight (kg)", 30, 200, 75)
height = st.sidebar.number_input("Height (cm)", 120, 250, 160)
age = st.sidebar.number_input("Age", 10, 100, 25)
sex = st.sidebar.selectbox("Sex", ['female', 'male'])
activity_level = st.sidebar.selectbox("Activity Level", ['sedentary', 'light', 'moderate', 'active', 'superactive'])

st.sidebar.subheader("🍽️ Foods You Like")
breakfast = st.sidebar.multiselect("Breakfast", df['Food'].unique())
lunch = st.sidebar.multiselect("Lunch", df['Food'].unique())
dinner = st.sidebar.multiselect("Dinner", df['Food'].unique())

st.sidebar.subheader("🚫 Foods You Dislike")
dislikes = st.sidebar.multiselect("Avoid These", df['Food'].unique())

# --- Generate Initial Plan ---
if st.sidebar.button("✨ Generate Meal Plan"):
    tdee = calculate_tdee(weight, height, age, sex, activity_level)
    daily_calories = tdee - 500
    preferences = {'breakfast': breakfast, 'lunch': lunch, 'dinner': dinner}
    plan = generate_meal_plan(preferences, daily_calories, dislikes)
    st.session_state.meal_plan = plan
    filename = f"mealplans_{st.session_state.username}.csv"
    plan.to_csv(filename, index=False)
    st.success("Meal plan generated and saved!")

# --- Display Meal Plan ---
if st.session_state.meal_plan is not None:
    st.subheader("📅 Your 7-Day Meal Plan")
    st.dataframe(st.session_state.meal_plan, use_container_width=True)

# --- Reshuffle Trigger ---
if st.button("🔄 Reshuffle Plan"):
    st.session_state.reshuffle_mode = True

if st.session_state.reshuffle_mode and st.session_state.meal_plan is not None:
    st.markdown("### 🔄 Reshuffle Options")
    reshuffle_option = st.radio("Choose Reshuffle Type:", ["Select parts to reshuffle", "Reshuffle entire plan"])

    if reshuffle_option == "Select parts to reshuffle":
        days = st.multiselect("Select day(s) to reshuffle", st.session_state.meal_plan["Day"].tolist())
        meals = st.multiselect("Which meals to reshuffle?", ["Breakfast", "Lunch", "Dinner"])
        more_dislikes = st.multiselect("Additional foods to avoid", df['Food'].unique())

        if st.button("✨ Confirm Partial Reshuffle"):
            prefs = {'breakfast': breakfast, 'lunch': lunch, 'dinner': dinner}
            updated_dislikes = list(set(dislikes + more_dislikes))
            new_partial = generate_meal_plan(prefs, daily_calories, updated_dislikes)
            for day in days:
                idx = st.session_state.meal_plan[st.session_state.meal_plan["Day"] == day].index[0]
                new_day = new_partial[new_partial["Day"] == day]
                for meal in meals:
                    st.session_state.meal_plan.at[idx, meal] = new_day.iloc[0][meal]
                st.session_state.meal_plan.at[idx, "Total Portion (g)"] = new_day.iloc[0]["Total Portion (g)"]
            st.session_state.meal_plan.to_csv(f"mealplans_{st.session_state.username}.csv", index=False)
            st.session_state.reshuffle_mode = False
            st.success("Plan updated!")
            st.experimental_rerun()

    elif reshuffle_option == "Reshuffle entire plan":
        new_dislikes_full = st.multiselect("Foods to avoid in new plan", df['Food'].unique())
        if st.button("✨ Confirm Full Reshuffle"):
            prefs = {'breakfast': breakfast, 'lunch': lunch, 'dinner': dinner}
            updated_dislikes = list(set(dislikes + new_dislikes_full))
            new_plan = generate_meal_plan(prefs, daily_calories, updated_dislikes)
            st.session_state.meal_plan = new_plan
            st.session_state.meal_plan.to_csv(f"mealplans_{st.session_state.username}.csv", index=False)
            st.session_state.reshuffle_mode = False
            st.success("Full plan reshuffled!")
            st.experimental_rerun()

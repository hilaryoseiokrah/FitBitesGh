# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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

# --- Neural Net Model ---
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

# --- Sidebar Inputs ---
st.sidebar.header("📋 Your Details")
weight = st.sidebar.number_input("Current Weight (kg)", 30, 200, 90)
target_weight = st.sidebar.number_input("Target Weight (kg)", 30, 200, 75)
height = st.sidebar.number_input("Height (cm)", 120, 250, 160)
age = st.sidebar.number_input("Age", 10, 100, 25)
sex = st.sidebar.selectbox("Sex", ['female', 'male'])
activity_level = st.sidebar.selectbox("Activity Level", ['sedentary', 'light', 'moderate', 'active', 'superactive'])

st.sidebar.subheader("🍽️ Foods You Like")
breakfast = st.sidebar.multiselect("Breakfast Options", df['Food'].unique())
lunch = st.sidebar.multiselect("Lunch Options", df['Food'].unique())
dinner = st.sidebar.multiselect("Dinner Options", df['Food'].unique())

st.sidebar.subheader("🚫 Foods You Dislike")
dislikes = st.sidebar.multiselect("Select Foods You Don't Want in Your Plan", df['Food'].unique())

# --- Main UI ---
st.title("🥗 Welcome to FitBites!")
st.markdown("""
**Your Personalized Ghanaian Meal Plan Assistant** 🇬🇭 🍛  
Ready to help you hit your weight goal in a sustainable, healthy, and delicious way!  
Please fill out your details and food preferences on the left, then click **Generate Meal Plan**.
""")

if st.sidebar.button("✨ Generate Meal Plan"):
    bmi = calculate_bmi(weight, height)
    tdee = calculate_tdee(weight, height, age, sex, activity_level)
    months = int(np.round(abs(weight - target_weight) / 2))
    daily_calories = tdee - 500

    st.success(f"🎯 Target Weight: {target_weight} kg")
    st.info(f"Estimated time to achieve goal: **{months} months**")
    st.info(f"🔥 Daily Calorie Target: **{daily_calories:.0f} kcal/day**")

    preferences = {
        'breakfast': breakfast,
        'lunch': lunch,
        'dinner': dinner
    }

    meal_plan = generate_meal_plan(preferences, daily_calories, dislikes=dislikes)

    st.subheader("📅 Your 7-Day Meal Plan")
    st.dataframe(meal_plan, use_container_width=True)

    # --- Reshuffle Section ---
    st.markdown("### 🔄 Not feeling this plan?")
    reshuffle_option = st.radio(
        "Would you like to reshuffle the entire plan or just parts of it?",
        ["Select parts to reshuffle", "Reshuffle entire plan"]
    )

    if reshuffle_option == "Select parts to reshuffle":
        days_to_reshuffle = st.multiselect(
            "Which day(s) would you like to reshuffle?",
            options=meal_plan["Day"].tolist(),
            default=meal_plan["Day"].tolist()
        )

        meals_to_reshuffle = st.multiselect(
            "Which meal(s)?",
            options=["Breakfast", "Lunch", "Dinner"],
            default=["Breakfast", "Lunch", "Dinner"]
        )

        new_dislikes = st.multiselect(
            "Any additional foods to exclude this time?",
            options=df['Food'].unique()
        )

        if st.button("✨ Confirm Partial Reshuffle"):
            updated_dislikes = list(set(dislikes + new_dislikes))
            partial_prefs = {m.lower(): preferences[m.lower()] for m in meals_to_reshuffle}
            reshuffled = generate_meal_plan(partial_prefs, daily_calories, dislikes=updated_dislikes)
            reshuffled = reshuffled[reshuffled["Day"].isin(days_to_reshuffle)]
            meal_plan_updated = meal_plan.copy()
            for _, row in reshuffled.iterrows():
                day_idx = meal_plan_updated[meal_plan_updated["Day"] == row["Day"]].index[0]
                for col in ["Breakfast", "Lunch", "Dinner", "Total Portion (g)"]:
                    if col.lower() in [m.lower() for m in meals_to_reshuffle] or col == "Total Portion (g)":
                        meal_plan_updated.at[day_idx, col] = row[col]
            st.subheader("📅 Your Updated 7-Day Meal Plan")
            st.dataframe(meal_plan_updated, use_container_width=True)

    elif reshuffle_option == "Reshuffle entire plan":
        new_dislikes_full = st.multiselect(
            "Want to exclude additional foods before reshuffling the whole plan?",
            options=df['Food'].unique()
        )
        if st.button("✨ Confirm Full Reshuffle"):
            updated_dislikes_full = list(set(dislikes + new_dislikes_full))
            meal_plan_full = generate_meal_plan(preferences, daily_calories, dislikes=updated_dislikes_full)
            st.subheader("📅 Your New Full 7-Day Meal Plan")
            st.dataframe(meal_plan_full, use_container_width=True)

else:
    st.warning("👉 Please complete your details and click **Generate Meal Plan** to begin!")

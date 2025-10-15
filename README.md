E-commerce Product Recommender
Project Overview

This project implements an E-commerce Product Recommender system that combines recommendation algorithms (content-based and collaborative filtering) with LLM-powered explanations. Users receive personalized product recommendations along with natural-language explanations describing why each product is suggested.

Objective:

Input: Product catalog + user behavior

Output: Recommended products + LLM-generated explanation (“Why this product?”)

Optional: Frontend dashboard for user interaction

Features

User-based recommendations: Collaborative filtering for user-product interactions.

Content-based recommendations: Uses product metadata (title, category, brand, attributes) with TF-IDF & cosine similarity.

LLM explanations: Provides readable reasoning for each recommendation.

Frontend: Streamlit dashboard to enter user ID and view recommendations.

Folder Structure
Ecommerce-Recommender/
│
├─ backend/                  # FastAPI backend
│   ├─ app/
│   │   ├─ main.py           # API entrypoint
│   │   ├─ recommender/
│   │   │   ├─ item_cf.py    # Collaborative filtering
│   │   │   └─ content_based.py  # Content-based & LLM scoring
│   │   └─ data/             # Products & interactions CSV
│
├─ frontend/                 # Streamlit frontend
│   └─ app.py
│
├─ requirements.txt          # Python dependencies
└─ README.md                 # This file

Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/<your-username>/Ecommerce-Recommender.git
cd Ecommerce-Recommender

2️⃣ Create and activate virtual environment

Windows (PowerShell):

python -m venv venv
.\venv\Scripts\Activate.ps1


Linux / Mac:

python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run backend (FastAPI)
cd backend
uvicorn main:app --reload


Backend runs at: http://127.0.0.1:8000

Swagger docs: http://127.0.0.1:8000/docs

5️⃣ Run frontend (Streamlit)
cd frontend
streamlit run app.py


Frontend opens at: http://localhost:8501

Usage

Enter a user ID in the frontend.

Click Submit to view recommended products.

Each recommendation shows:

Product details

Score (relevance)

LLM-generated explanation

Demo Video

Watch demo

# E-commerce Product Recommender

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-frontend-green)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-red)](https://fastapi.tiangolo.com/)

## Project Overview

This project implements an **E-commerce Product Recommender** system combining:

- **Collaborative filtering** and **content-based recommendation**
- **LLM-powered explanations** for personalized recommendations

**Objective:**

- **Input:** Product catalog + user behavior
- **Output:** Recommended products with natural-language explanations
- **Optional:** Frontend dashboard for user interaction

## Features

- **User-based recommendations** using collaborative filtering
- **Content-based recommendations** using product metadata (title, category, brand, attributes) with TF-IDF & cosine similarity
- **LLM explanations:** Provides readable reasoning for each recommendation
- **Frontend:** Streamlit dashboard to input user ID and display recommendations

bash
Copy code

## Setup Instructions

## 1. Clone the repository

git clone https://github.com/<your-username>/Ecommerce-Recommender.git

cd Ecommerce-Recommender

##2. Create and activate virtual environment

python -m venv venv

.\venv\Scripts\Activate.ps1

##3. Install dependencies

pip install -r requirements.txt

##4. Run Backend (FastAPI)

cd backend

uvicorn main:app --reload

Backend URL: http://127.0.0.1:8000

Swagger docs: http://127.0.0.1:8000/docs

##5. Run Frontend (Streamlit)

cd frontend

streamlit run app.py

Frontend URL: http://localhost:8501

##How It Works

Collaborative Filtering: Scores products based on user-product interaction matrix

Content-Based Scoring: TF-IDF vectorization of product metadata and cosine similarity with user profile

LLM Explanation: Generates human-readable reasoning for recommended products


##Deliverables

Backend API for recommendations

Frontend Dashboard for user interaction

LLM-powered explanations



###Evaluation Focus

Recommendation accuracy

LLM explanation quality

Code design and structure

Documentation and setup instructions

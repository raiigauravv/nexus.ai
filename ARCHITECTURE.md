# NEXUS-AI Architecture Document

## Overview
**NEXUS-AI** is a production-grade Enterprise ML platform offering an orchestration of multiple machine learning models and capabilities in a unified, visually stunning interface ("Hollow Purple" aesthetic).

It transitioned from a synthetic demo to a real-world enterprise prototype through several phases of architectural upgrades, substituting mock heuristics with trained models on real datasets.

## Core Modules & Upgrades

### 1. Real-Time Fraud Detection (XGBoost)
* **Objective:** Replace rule-based heuristics with a high-performance predictive model.
* **Dataset:** Kaggle Credit Card Fraud Detection dataset (284k transactions).
* **Technique:** Handled the 0.17% fraud class imbalance using **SMOTE** (Synthetic Minority Over-sampling). Trained an **XGBoost Classifier**.
* **Performance:** AUC-ROC of 0.98. Integrated with SHAP-like feature importance values natively displayed in the UI.

### 2. NLP Sentiment Pipeline (RoBERTa)
* **Objective:** Upgrade from a basic DistilBERT/VADER hybrid to a state-of-the-art context-aware model.
* **Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`.
* **Implementation:** Integrated huggingface transformers into the FastAPI backend. Analyzes subjectivity, nuance, and internet slang to return robust 3-class (Positive, Neutral, Negative) probability distributions.

### 3. Recommendation Engine (SVD)
* **Objective:** Move from 500 synthetic users and 20 dummy products to a massive real-world interaction matrix.
* **Dataset:** Amazon Cell Phones and Accessories Reviews (Kaggle).
* **Technique:** Processed 67k+ reviews across 47,000+ real users and 720 products. Extracted sparse matrices and performed **Truncated Singular Value Decomposition (SVD)** (rank 50) to build latent feature embeddings for Collaborative Filtering.
* **Features:** Personalized "For You" feeds derived from matrix multiplication, and a Bayesian average "Trending" algorithm for global popularity.

### 4. Computer Vision (CLIP)
* **Model:** OpenAI's `CLIP` (ViT-B/32).
* **Integration:** Reindexed the entire 720-product Amazon catalog. The visual search feature encodes user-uploaded images and performs a Pinecone vector similarity search in a 512-dimensional embedding space to find visually similar items.

### 5. Document Q&A / Intelligent Agent
* **Integration:** Powered by the **Gemini 1.5** model via the official API.
* **Capabilities:** Acts as a centralized orchestration router. Seamlessly transitions between chatting, extracting context from uploaded PDFs via RAG, and querying other internal microservices.

### 6. Authentication & Security
* **System:** JSON Web Tokens (JWT).
* **Database:** PostgreSQL storing user profiles and bcrypt-hashed passwords.
* **Implementation:** The frontend implements secure `React Portals` for the Auth modals to prevent stacking context clipping, ensuring a perfect aesthetic.

## Stack
* **Frontend:** Next.js 14 (App Router), React, Tailwind CSS, Framer Motion, Lucide Icons.
* **Backend:** FastAPI, SQLAlchemy, Uvicorn.
* **Database:** PostgreSQL, Redis (Caching), MinIO (Object Storage).
* **ML Ops:** MLflow (Tracking), Kafka (Event Streaming).
* **Deployment:** Fully dockerized via `docker-compose`.

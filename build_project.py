#!/usr/bin/env python3
"""
Master build script for INSY669 GLP-1 Text Analytics Project.
Generates realistic synthetic data, runs all analysis, saves figures, and creates notebooks.
"""

import pandas as pd
import numpy as np
import random
import os
import json
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

BASE_DIR = "/home/ubuntu/classes/insy669/final-project"
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# 1. GENERATE REALISTIC DATA
# ============================================================

# --- Reddit posts ---
reddit_templates_positive = [
    "I've been on {drug} for {months} months and lost {lbs} lbs! Best decision ever.",
    "Week {weeks} on {drug}. Down {lbs} pounds. The nausea was rough at first but totally worth it.",
    "{drug} changed my life. I finally feel like myself again after losing {lbs} lbs.",
    "Started {drug} at {start_dose}mg and just moved up. Already seeing amazing results!",
    "My A1C dropped from {a1c_high} to {a1c_low} on {drug}. Doctor is thrilled.",
    "3 months on {drug} and my relationship with food has completely changed. No more binge eating.",
    "Insurance finally approved {drug}! Starting next week. So excited based on everyone's results here.",
    "Lost {lbs} lbs on {drug} in {months} months. Blood pressure normalized too!",
    "The appetite suppression from {drug} is incredible. I actually forget to eat now.",
    "{drug} has been a game changer for my PCOS symptoms. Weight coming off steadily.",
    "Finally found a pharmacy with {drug} in stock! Compounding pharmacy saved me.",
    "My endocrinologist recommended {drug} and I'm so glad I listened. {lbs} lbs down!",
    "PSA: {drug} works even better with light exercise. Walking 30 min/day + {drug} = amazing results.",
    "Went from size {size_high} to size {size_low} thanks to {drug}. Can't believe the transformation.",
    "The food noise is GONE on {drug}. This is what normal people must feel like around food.",
]

reddit_templates_negative = [
    "The nausea on {drug} is unbearable. I can barely keep water down.",
    "Had to stop {drug} due to severe gastroparesis. My stomach just won't empty.",
    "Anyone else getting horrible sulfur burps on {drug}? It's so embarrassing.",
    "I'm on {drug} and the constipation is REAL. Nothing helps.",
    "Spent $1200/month on {drug} because insurance won't cover it. This is insane.",
    "{drug} shortage is killing me. Can't find it anywhere in my area.",
    "Lost hair on {drug}. Nobody warned me about this side effect.",
    "The injection site reactions from {drug} are getting worse. Red welts every time.",
    "Gained all the weight back after stopping {drug}. Feeling so discouraged.",
    "Severe fatigue on {drug}. Can barely function at work.",
    "My doctor won't prescribe {drug} because my BMI is 'only' 28. So frustrated.",
    "The cost of {drug} without insurance is criminal. $1300 for a month supply.",
    "Pancreatitis scare while on {drug}. ER visit and had to discontinue immediately.",
    "I feel like a completely different person on {drug} and not in a good way. Anxiety through the roof.",
    "Gallbladder issues after 6 months on {drug}. Had to get it removed.",
]

reddit_templates_neutral = [
    "Starting {drug} tomorrow. Any tips for managing side effects?",
    "What dose of {drug} is everyone on? I'm at {dose}mg weekly.",
    "Has anyone switched from {drug} to {other_drug}? Wondering about differences.",
    "How long did it take for {drug} to start working for you?",
    "Does {drug} need to be refrigerated after first use?",
    "Question about {drug} and alcohol - is it safe to drink occasionally?",
    "Looking for {drug} in the {city} area. Any leads on pharmacies that have it?",
    "What's the difference between {drug} and the compounded semaglutide?",
    "My doctor wants me to try {drug} but I'm nervous about needles.",
    "Comparing {drug} vs tirzepatide. Anyone tried both?",
]

drugs = ["Ozempic", "Wegovy"]
other_drugs = {"Ozempic": "Wegovy", "Wegovy": "Ozempic"}
subreddits = ["r/Ozempic", "r/Semaglutide", "r/WegovyWeightLoss"]
cities = ["Chicago", "NYC", "LA", "Houston", "Phoenix", "Dallas", "Miami", "Atlanta"]

reddit_posts = []
for i in range(800):
    drug = random.choice(drugs)
    subreddit = random.choice(subreddits)
    roll = random.random()
    if roll < 0.45:
        template = random.choice(reddit_templates_positive)
        text = template.format(
            drug=drug, months=random.randint(1,12), lbs=random.randint(10,80),
            weeks=random.randint(1,52), start_dose=random.choice([0.25, 0.5, 1.0]),
            a1c_high=round(random.uniform(7.5, 11.0), 1),
            a1c_low=round(random.uniform(5.5, 7.0), 1),
            size_high=random.randint(16, 24), size_low=random.randint(8, 14),
            dose=random.choice([0.25, 0.5, 1.0, 1.7, 2.4]),
            other_drug=other_drugs[drug], city=random.choice(cities)
        )
    elif roll < 0.75:
        template = random.choice(reddit_templates_negative)
        text = template.format(
            drug=drug, months=random.randint(1,12), lbs=random.randint(10,80),
            weeks=random.randint(1,52), start_dose=random.choice([0.25, 0.5, 1.0]),
            dose=random.choice([0.25, 0.5, 1.0, 1.7, 2.4]),
            other_drug=other_drugs[drug], city=random.choice(cities)
        )
    else:
        template = random.choice(reddit_templates_neutral)
        text = template.format(
            drug=drug, months=random.randint(1,12), lbs=random.randint(10,80),
            weeks=random.randint(1,52), start_dose=random.choice([0.25, 0.5, 1.0]),
            dose=random.choice([0.25, 0.5, 1.0, 1.7, 2.4]),
            other_drug=other_drugs[drug], city=random.choice(cities)
        )
    
    date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
    reddit_posts.append({
        "id": f"reddit_{i}",
        "source": "reddit",
        "subreddit": subreddit,
        "text": text,
        "date": date.strftime("%Y-%m-%d"),
        "score": random.randint(1, 500),
        "num_comments": random.randint(0, 150),
        "drug_mentioned": drug,
    })

df_reddit = pd.DataFrame(reddit_posts)
df_reddit.to_csv(os.path.join(DATA_DIR, "reddit_posts.csv"), index=False)
print(f"Generated {len(df_reddit)} Reddit posts")

# --- WebMD reviews ---
webmd_positive = [
    "I have been taking {drug} for {months} months for weight management. I have lost {lbs} pounds with minimal side effects. Highly recommend discussing with your doctor.",
    "{drug} has significantly reduced my appetite. I no longer obsess over food and have lost {lbs} lbs. The once-weekly injection is very convenient.",
    "After struggling with obesity for years, {drug} finally helped me lose weight. Down {lbs} lbs in {months} months. Some nausea initially but it passed.",
    "Excellent medication. {drug} helped me lose {lbs} pounds and my blood sugar is now under control. Worth every penny.",
    "Life changing! {drug} reduced my food noise completely. I eat reasonable portions now and have lost {lbs} lbs.",
]

webmd_negative = [
    "Terrible experience with {drug}. Constant nausea, vomiting, and diarrhea. Had to discontinue after {months} months.",
    "{drug} caused severe stomach pain and I ended up in the ER. Would not recommend.",
    "Lost some weight on {drug} but the side effects were not worth it. Sulfur burps, constipation, and fatigue daily.",
    "Cannot afford {drug}. My insurance denied coverage and the out of pocket cost is over $1000 per month. Healthcare system is broken.",
    "{drug} gave me pancreatitis. I was hospitalized for a week. Be very careful with this medication.",
]

webmd_mixed = [
    "{drug} works for weight loss ({lbs} lbs in {months} months) but the GI side effects are significant. You need to weigh the pros and cons.",
    "Some weight loss on {drug} but plateaued after {months} months. Nausea comes and goes. Not sure if I'll continue.",
    "{drug} helped with appetite but I get terrible headaches. Lost {lbs} lbs though so trying to push through.",
]

webmd_reviews = []
for i in range(300):
    drug = random.choice(drugs)
    roll = random.random()
    if roll < 0.5:
        template = random.choice(webmd_positive)
        rating = random.choice([4, 5])
    elif roll < 0.8:
        template = random.choice(webmd_negative)
        rating = random.choice([1, 2])
    else:
        template = random.choice(webmd_mixed)
        rating = 3
    
    text = template.format(drug=drug, months=random.randint(1,12), lbs=random.randint(5,60))
    date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
    
    webmd_reviews.append({
        "id": f"webmd_{i}",
        "source": "webmd",
        "drug": drug,
        "text": text,
        "rating": rating,
        "date": date.strftime("%Y-%m-%d"),
        "condition": random.choice(["Weight Loss", "Type 2 Diabetes", "Obesity"]),
    })

df_webmd = pd.DataFrame(webmd_reviews)
df_webmd.to_csv(os.path.join(DATA_DIR, "webmd_reviews.csv"), index=False)
print(f"Generated {len(df_webmd)} WebMD reviews")

# --- News articles ---
news_templates = [
    "New study finds {drug} leads to average {pct}% body weight reduction in clinical trial involving {n} participants. Researchers at {institution} report significant improvements in cardiovascular outcomes alongside weight loss.",
    "The FDA has expanded the approved use of {drug} for chronic weight management in adults with obesity. Health officials say the drug represents a major advancement in obesity treatment options.",
    "Novo Nordisk reports {drug} shortage easing as production capacity increases. The pharmaceutical company has invested ${billions} billion in new manufacturing facilities to meet unprecedented demand.",
    "Concerns raised about long-term safety of GLP-1 receptor agonists like {drug}. A panel of endocrinologists calls for more comprehensive post-market surveillance studies.",
    "Wall Street analysts project {drug} sales to reach ${revenue} billion by 2025, making it one of the best-selling drugs in pharmaceutical history. Novo Nordisk stock hits all-time high.",
    "Celebrity endorsements drive surge in {drug} prescriptions for weight loss, raising ethical concerns among healthcare professionals about off-label use and medication shortages for diabetic patients.",
    "{drug} covered by Medicare for first time under new policy. Advocates say this represents a critical step in recognizing obesity as a chronic disease requiring medical treatment.",
    "Insurance companies increasingly covering {drug} for weight loss as evidence mounts for long-term health benefits including reduced cardiovascular risk and improved metabolic markers.",
    "Counterfeit {drug} found in multiple states, prompting FDA warning. Authorities seize thousands of fake injection pens containing potentially dangerous substances from unauthorized sellers.",
    "New research suggests {drug} may have benefits beyond weight loss, including reduced risk of kidney disease, heart failure, and certain types of cancer. Scientists call findings 'promising but preliminary'.",
    "The rise of GLP-1 drugs like {drug} is reshaping the food industry. Major food companies report declining snack sales as millions of Americans reduce caloric intake on these medications.",
    "Doctors debate whether patients should stay on {drug} indefinitely. Weight regain after discontinuation raises questions about the long-term treatment paradigm for obesity management.",
    "Global demand for {drug} outstrips supply as countries beyond the US see rising obesity rates. WHO calls for equitable access to effective obesity treatments worldwide.",
    "{drug} manufacturer faces lawsuits over alleged failure to warn about gastroparesis and other serious gastrointestinal side effects. Plaintiffs claim permanent digestive damage.",
    "Telehealth platforms make it easier than ever to get {drug} prescriptions, but critics warn about inadequate medical oversight and the risks of prescribing weight loss drugs without thorough evaluation.",
    "Pediatric obesity experts discuss potential role of {drug} in treating adolescent obesity after promising trial results show significant weight reduction in teenagers aged 12-17.",
    "Economic analysis shows {drug} could save healthcare system billions by reducing obesity-related conditions including type 2 diabetes, hypertension, and joint replacement surgeries.",
    "Social media influencers fuel {drug} trend, posting dramatic before-and-after photos. Mental health experts express concern about unrealistic expectations and body image pressures.",
    "Compounding pharmacies offer cheaper semaglutide alternatives to brand-name {drug}, sparking debate about safety, efficacy, and FDA regulation of compounded medications.",
    "New competitor drugs challenge {drug} dominance in the GLP-1 market. Eli Lilly's tirzepatide shows comparable or superior weight loss results in head-to-head clinical trials.",
]

news_sources = ["Reuters", "CNN Health", "The New York Times", "STAT News", "Medical News Today", 
                "NPR Health", "The Washington Post", "Bloomberg", "NBC News Health", "Associated Press",
                "Forbes Health", "The Wall Street Journal", "BBC Health", "CNBC", "Healthline"]

news_articles = []
for i in range(400):
    drug = random.choice(drugs)
    template = random.choice(news_templates)
    text = template.format(
        drug=drug, pct=round(random.uniform(12, 22), 1), n=random.randint(500, 5000),
        institution=random.choice(["Harvard Medical School", "Mayo Clinic", "Johns Hopkins", "Cleveland Clinic", "Stanford Medicine"]),
        billions=round(random.uniform(2, 8), 1), revenue=random.randint(15, 30),
    )
    date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 365))
    
    news_articles.append({
        "id": f"news_{i}",
        "source": random.choice(news_sources),
        "text": text,
        "date": date.strftime("%Y-%m-%d"),
        "drug_mentioned": drug,
        "category": random.choice(["Health", "Business", "Science", "Regulation", "Society"]),
    })

df_news = pd.DataFrame(news_articles)
df_news.to_csv(os.path.join(DATA_DIR, "news_articles.csv"), index=False)
print(f"Generated {len(df_news)} news articles")

print("\n=== Data generation complete ===")
print(f"Total public corpus: {len(df_reddit) + len(df_webmd)} documents")
print(f"Total media corpus: {len(df_news)} documents")

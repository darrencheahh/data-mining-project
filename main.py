import pandas as pd
import re
import spacy
import openai
from sklearn.feature_extraction.text import CountVectorizer

openai.api_key = 'sk-proj-hOCv06CKuBr6Nxwqk5QCT3BlbkFJ2yikQYlvSjCozO1ekEXI'

#Load dataset
file_path = 'cnbc_headlines02.csv'
finance_info = pd.read_csv(file_path)

# Clean dataset
finance_info.dropna(subset=['Time'], inplace=True)

# Strip the timezone information before converting to datetime
finance_info['Time'] = finance_info['Time'].apply(lambda x: re.sub(r'\s+(AM|PM)', r' \1', x.strip()))

# Now convert to datetime
try:
    finance_info['Date'] = pd.to_datetime(finance_info['Time'], format='%I:%M %p ET %a, %d %B %Y')
except Exception as e:
    print(f"Error parsing datetime: {e}")

nlp = spacy.load('en_core_web_sm')

# Preprocess text
def preprocess_headline(text):
    doc = nlp(text)
    # Generate a list of lemma
    simplified_headlines = " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_stop])
    return simplified_headlines

# Apply preprocessing to headlines
finance_info['simplified_headlines'] = finance_info['Headlines'].apply(preprocess_headline)

print(finance_info['simplified_headlines'].unique()[:30])









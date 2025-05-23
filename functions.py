import pandas as pd
import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import pos_tag, word_tokenize
import textstat
import pickle
import logging
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("answer-sheet-evaluation")

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Load model once and cache it
@lru_cache(maxsize=1)
def get_sentence_transformer_model():
    logger.info("Loading sentence transformer model")
    return SentenceTransformer('all-mpnet-base-v2')

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF, using OCR if needed."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # If page has no text, apply OCR
            if not text.strip():
                logger.info(f"Using OCR for page {page_num+1} in {os.path.basename(pdf_path)}")
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
            
            full_text += text + "\n"
        
        return full_text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to process PDF: {str(e)}")

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    cleaned_text = re.sub(r'[^a-zA-Z0-9.,!? ]', '', text)
    return cleaned_text

def extract_questions_and_answers(text: str) -> pd.DataFrame:
    """Extract questions and answers from text using regex pattern matching."""
    pattern = r'Question\s+(\d+):\s+(.*?)(?:\s+Answer:\s+(.*?))?(?=Question\s+\d+:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        logger.warning("No questions and answers found in the text")
        return pd.DataFrame(columns=['ID', 'Question', 'Answer'])
    
    items = []
    for match in matches:
        question_id = match[0].strip()
        question_text = match[1].strip()
        answer_text = match[2].strip() if len(match) > 2 and match[2].strip() else "No answer provided"
        
        items.append({
            'ID': question_id,
            'Question': question_text,
            'Answer': answer_text
        })
    
    return pd.DataFrame(items)

def string_similarity(a: str, b: str) -> float:
    """Calculate string similarity using SequenceMatcher."""
    return SequenceMatcher(None, a, b).ratio()

def create_comparison_dataframe(reference_df: pd.DataFrame, student_df: pd.DataFrame) -> pd.DataFrame:
    """Create a comparison dataframe between reference and student answers."""
    if reference_df.empty:
        logger.warning("Reference dataframe is empty")
        return pd.DataFrame(columns=['ID', 'Question', 'Reference Answer', 'Student Answer'])
    
    results = []
    
    for _, ref_row in reference_df.iterrows():
        ref_id = ref_row['ID']
        ref_question = ref_row['Question']
        ref_answer = ref_row['Answer']
        
        # Try to find matching question by ID
        student_row = student_df[student_df['ID'] == ref_id]
        
        if not student_row.empty:
            student_answer = student_row['Answer'].values[0]
        else:
            # Try to find a matching question by similarity
            best_match = None
            best_score = 0
            
            for _, stu_row in student_df.iterrows():
                score = string_similarity(ref_question, stu_row['Question'])
                if score > best_score:
                    best_score = score
                    best_match = stu_row
            
            if best_match is not None and best_score > 0.8:
                student_answer = best_match['Answer']
                logger.info(f"Found matching question for ID {ref_id} with similarity score {best_score:.2f}")
            else:
                student_answer = "No answer provided"
                logger.warning(f"No matching answer found for question ID {ref_id}")
        
        results.append({
            'ID': ref_id,
            'Question': ref_question,
            'Reference Answer': ref_answer,
            'Student Answer': student_answer
        })
    
    return pd.DataFrame(results)

def get_pos_features(text: str) -> Dict[str, float]:
    """Extract part-of-speech features from text."""
    try:
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        
        total_words = len(tokens)
        
        if total_words == 0:
            return {
                'noun_ratio': 0,
                'verb_ratio': 0,
                'adj_ratio': 0,
                'adv_ratio': 0,
                'pos_diversity': 0
            }
        
        return {
            'noun_ratio': (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) +
                          pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / total_words,
            'verb_ratio': (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) +
                          pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) +
                          pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / total_words,
            'adj_ratio': (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) +
                         pos_counts.get('JJS', 0)) / total_words,
            'adv_ratio': (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) +
                         pos_counts.get('RBS', 0)) / total_words,
            'pos_diversity': len(pos_counts) / total_words
        }
    except Exception as e:
        logger.error(f"Error calculating POS features: {str(e)}")
        return {
            'noun_ratio': 0,
            'verb_ratio': 0,
            'adj_ratio': 0,
            'adv_ratio': 0,
            'pos_diversity': 0
        }

def get_readability_scores(reference_answer: str, student_answer: str) -> Dict[str, float]:
    """Calculate readability scores for answers."""
    try:
        ref_fk = textstat.flesch_kincaid_grade(reference_answer)
        student_fk = textstat.flesch_kincaid_grade(student_answer)
        
        # Avoid division by zero
        if ref_fk == 0:
            return {"flesch_kincaid_ratio": 0}
        
        return {"flesch_kincaid_ratio": student_fk / ref_fk}
    except Exception as e:
        logger.error(f"Error calculating readability scores: {str(e)}")
        return {"flesch_kincaid_ratio": 0}

def process_answer_sheet_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process answer sheet data and extract features for scoring."""
    if df.empty:
        logger.warning("Empty dataframe provided for processing")
        return df
    
    df = df.copy()
    
    # Get the model
    model = get_sentence_transformer_model()
    
    # Encode texts
    ref_embeddings = model.encode(df['Reference Answer'].tolist())
    stu_embeddings = model.encode(df['Student Answer'].tolist())
    que_embeddings = model.encode(df['Question'].tolist())
    
    # Calculate semantic similarity
    df['Semantic Similarity'] = cosine_similarity(ref_embeddings, stu_embeddings).diagonal()
    
    # Calculate length ratio
    df['Length Ratio'] = df.apply(
        lambda row: len(row['Student Answer']) / len(row['Reference Answer']) 
        if len(row['Reference Answer']) > 0 else 0, 
        axis=1
    )
    
    # Calculate question-student similarity
    df['question_student_similarity'] = cosine_similarity(
        np.stack(que_embeddings), np.stack(stu_embeddings)
    ).diagonal()
    
    # Calculate POS features
    df['Student_POS_Features'] = df['Student Answer'].apply(get_pos_features)
    df['Reference_POS_Features'] = df['Reference Answer'].apply(get_pos_features)
    
    pos_features = ['noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'pos_diversity']
    
    for feature in pos_features:
        df[f'POS_{feature}_diff'] = (
            df['Student_POS_Features'].apply(lambda x: x[feature]) - 
            df['Reference_POS_Features'].apply(lambda x: x[feature])
        )
    
    # Calculate POS similarity
    df['POS_similarity'] = 1 - df[[f'POS_{f}_diff' for f in pos_features]].abs().sum(axis=1) / 5
    
    # Calculate readability ratio
    df["flesch_kincaid_ratio"] = df.apply(
        lambda row: get_readability_scores(row["Reference Answer"], row["Student Answer"])["flesch_kincaid_ratio"], 
        axis=1
    )
    
    features = [
        'Semantic Similarity', 'Length Ratio', 'question_student_similarity',
        'POS_noun_ratio_diff', 'POS_similarity', 'flesch_kincaid_ratio'
    ]
    
    return df[['ID', 'Question', 'Reference Answer', 'Student Answer'] + features]

def predict_score(new_data: pd.DataFrame, model_path: str, scaler_path: str) -> np.ndarray:
    """Predict scores using the trained model."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        features = [
            'Semantic Similarity', 'Length Ratio', 'question_student_similarity',
            'POS_noun_ratio_diff', 'POS_similarity', 'flesch_kincaid_ratio'
        ]
        
        X_scaled = scaler.transform(new_data[features])
        predictions = model.predict(X_scaled)
        
        return np.clip(np.round(predictions), 0, 10).astype(int)
    except Exception as e:
        logger.error(f"Error predicting scores: {str(e)}")
        raise ValueError(f"Failed to predict scores: {str(e)}")

def main(reference_pdf_path: str, student_pdf_path: str, model_path: str, scaler_path: str) -> pd.DataFrame:
    """Main function to process PDFs and evaluate answers."""
    logger.info(f"Processing reference PDF: {os.path.basename(reference_pdf_path)}")
    reference_text = extract_text_from_pdf(reference_pdf_path)
    
    logger.info(f"Processing student PDF: {os.path.basename(student_pdf_path)}")
    student_text = extract_text_from_pdf(student_pdf_path)
    
    logger.info("Extracting questions and answers")
    reference_df = extract_questions_and_answers(reference_text)
    student_df = extract_questions_and_answers(student_text)
    
    if reference_df.empty:
        logger.error("No questions found in reference PDF")
        raise ValueError("No questions found in reference PDF")
    
    logger.info("Creating comparison dataframe")
    comparison_df = create_comparison_dataframe(reference_df, student_df)
    
    logger.info("Processing answer sheet data")
    processed_df = process_answer_sheet_data(comparison_df)
    
    logger.info("Predicting scores")
    scores = predict_score(processed_df, model_path, scaler_path)
    
    processed_df['Predicted Score'] = scores
    
    logger.info(f"Evaluation complete. Processed {len(processed_df)} questions.")
    return processed_df
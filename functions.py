import pandas as pd
import re
import fitz # PyMuPDF
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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from datetime import datetime
import zipfile

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
    
    # ✅ FIXED: Include ALL 10 features in correct order
    features = [
        'Semantic Similarity', 
        'Length Ratio', 
        'question_student_similarity',
        'POS_noun_ratio_diff', 
        'POS_similarity', 
        'flesch_kincaid_ratio',
        'POS_pos_diversity_diff',
        'POS_verb_ratio_diff',
        'POS_adv_ratio_diff',
        'POS_adj_ratio_diff'
    ]
    
    return df[['ID', 'Question', 'Reference Answer', 'Student Answer'] + features]


def predict_score(new_data: pd.DataFrame, model_path: str, scaler_path: str) -> np.ndarray:
    """Predict scores using the trained model."""
    try:
        # ✅ FIXED: Use joblib instead of pickle
        import joblib
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # ✅ FIXED: Include ALL 10 features in correct order
        features = [
            'Semantic Similarity', 
            'Length Ratio', 
            'question_student_similarity',
            'POS_noun_ratio_diff', 
            'POS_similarity', 
            'flesch_kincaid_ratio',
            'POS_pos_diversity_diff',
            'POS_verb_ratio_diff',
            'POS_adv_ratio_diff',
            'POS_adj_ratio_diff'
        ]
        
        X_scaled = scaler.transform(new_data[features])
        predictions = model.predict(X_scaled)
        
        # ✅ FIXED: LogisticAT returns integers directly, no need to round
        return predictions.astype(int)
    
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

def generate_pdf_report(student_data: Dict, request_id: str) -> str:
    """Generate a PDF report for a single student evaluation."""
    try:
        # Create temp file for PDF
        pdf_path = f"temp/report_{request_id}_{student_data['student_name'].replace(' ', '_')}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        # Title
        title = Paragraph("Answer Sheet Evaluation Report", title_style)
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Student Information
        student_info = [
            ["Student Name:", student_data["student_name"]],
            ["Total Score:", f"{student_data['total_score']} out of {student_data['max_possible_score']}"],
            ["Percentage:", f"{student_data['percentage']}%"],
            ["Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        info_table = Table(student_info, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Question-wise breakdown
        story.append(Paragraph("Question-wise Breakdown", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        for idx, result in enumerate(student_data["results"], 1):
            # Question header
            question_text = f"Question {idx}: {result['Question']}"
            story.append(Paragraph(question_text, styles['Heading3']))
            story.append(Spacer(1, 6))
            
            # Student Answer (FULL TEXT - NO TRUNCATION)
            story.append(Paragraph("<b>Student Answer:</b>", styles['Normal']))
            story.append(Spacer(1, 3))
            
            # Use Paragraph for proper text wrapping of long answers
            answer_text = result["Student_Answer"] if result["Student_Answer"] else "No answer provided"
            answer_para = Paragraph(answer_text, styles['Normal'])
            story.append(answer_para)
            story.append(Spacer(1, 10))
            
            # Score and Feedback table
            score_feedback_data = [
                ["Score:", f"{result['Predicted_Score']}/10"],
                ["Feedback:", result["Feedback"]]
            ]
            
            score_table = Table(score_feedback_data, colWidths=[1.5*inch, 4.5*inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(score_table)
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Generated PDF report for {student_data['student_name']}")
        return pdf_path
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise ValueError(f"Failed to generate PDF report: {str(e)}")


def generate_zip_reports(evaluations: List[Dict], request_id: str) -> str:
    """Generate a ZIP file containing PDF reports for all students."""
    try:
        zip_path = f"temp/reports_{request_id}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for evaluation in evaluations:
                # Generate PDF for each student
                pdf_path = generate_pdf_report(evaluation, request_id)
                
                # Add PDF to ZIP
                safe_name = evaluation["student_name"].replace(" ", "_")
                zip_filename = f"{safe_name}_evaluation_report.pdf"
                zip_file.write(pdf_path, zip_filename)
                
                # Clean up individual PDF
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
            
            # Add summary report
            summary_data = generate_summary_report(evaluations)
            zip_file.writestr("class_summary.txt", summary_data)
        
        logger.info(f"Generated ZIP file with {len(evaluations)} reports")
        return zip_path
        
    except Exception as e:
        logger.error(f"Error generating ZIP reports: {str(e)}")
        raise ValueError(f"Failed to generate ZIP reports: {str(e)}")

def generate_summary_report(evaluations: List[Dict]) -> str:
    """Generate a text summary of all evaluations."""
    try:
        summary_lines = []
        summary_lines.append("CLASS EVALUATION SUMMARY REPORT")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Total Students: {len(evaluations)}")
        summary_lines.append("")
        
        # Calculate statistics
        scores = [eval_data["percentage"] for eval_data in evaluations]
        avg_score = sum(scores) / len(scores) if scores else 0
        highest_score = max(scores) if scores else 0
        lowest_score = min(scores) if scores else 0
        
        summary_lines.append("CLASS STATISTICS:")
        summary_lines.append(f"Average Score: {avg_score:.2f}%")
        summary_lines.append(f"Highest Score: {highest_score:.2f}%")
        summary_lines.append(f"Lowest Score: {lowest_score:.2f}%")
        summary_lines.append("")
        
        # Individual student results
        summary_lines.append("INDIVIDUAL RESULTS:")
        summary_lines.append("-" * 30)
        
        # Sort by percentage (highest first)
        sorted_evaluations = sorted(evaluations, key=lambda x: x["percentage"], reverse=True)
        
        for idx, evaluation in enumerate(sorted_evaluations, 1):
            summary_lines.append(f"{idx:2d}. {evaluation['student_name']:<20} - {evaluation['percentage']:6.2f}% ({evaluation['total_score']}/{evaluation['max_possible_score']})")
        
        summary_lines.append("")
        summary_lines.append("=" * 50)
        summary_lines.append("End of Report")
        
        return "\n".join(summary_lines)
        
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")
        return "Error generating summary report"

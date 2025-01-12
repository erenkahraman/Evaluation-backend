from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Initialize the semantic similarity model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the NLI model for entailment checking
nli_model_name = "roberta-large-mnli"
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

def normalize_text(text):
    # Basic text normalization
    text = text.lower().strip()
    # Remove multiple spaces
    text = ' '.join(text.split())
    return text

def calculate_score(similarity):
    """Convert similarity to a score between 0 and 10 using a lenient scaling."""
    similarity = max(0.0, min(1.0, similarity))
    if similarity >= 0.7:  # High similarity
        score = 7.0 + (similarity - 0.7) * (3.0 / 0.3)  
    elif similarity >= 0.4:  # Medium similarity
        score = 4.0 + (similarity - 0.4) * (3.0 / 0.3)  
    else:  # Low similarity
        score = similarity * (4.0 / 0.4)  
    return round(max(0.0, min(10.0, score)), 2)

def nli_entailment(premise, hypothesis):
    """Compute entailment probabilities using the NLI model."""
    inputs = nli_tokenizer.encode_plus(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = nli_model(**inputs)
    logits = outputs.logits.detach().numpy()[0]
    # Convert logits to probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    # The model's labels: 0 -> contradiction, 1 -> neutral, 2 -> entailment
    return {"contradiction": probabilities[0], 
            "neutral": probabilities[1], 
            "entailment": probabilities[2]}

def keyword_match_score(correct, student):
    """Calculate a simple keyword match score based on word overlap."""
    correct_words = set(correct.split())
    student_words = set(student.split())
    if not correct_words:
        return 0
    overlap = correct_words.intersection(student_words)
    # Scale overlap proportion to 0-10
    return (len(overlap) / len(correct_words)) * 10

def extract_numbers(text):
    """Extract numeric values from text."""
    return re.findall(r"\d+\.?\d*", text)

def numeric_check_score(correct, student):
    """Check numeric similarity between correct and student answers."""
    correct_nums = extract_numbers(correct)
    student_nums = extract_numbers(student)
    if not correct_nums:
        return 0
    matches = sum(1 for num in correct_nums if num in student_nums)
    # Scale match proportion to 0-10
    return (matches / len(correct_nums)) * 10

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        data = request.json
        student_answer = data.get('studentAnswer')
        correct_answer = data.get('correctAnswer')
        
        if not student_answer or not correct_answer:
            return jsonify({'error': 'Missing required fields'}), 400

        # Minimum length check (3 words)
        if len(student_answer.split()) < 3 or len(correct_answer.split()) < 3:
            return jsonify({
                'score': 0,
                'similarity': 0,
                'explanation': 'Answer too short (minimum 3 words required)'
            })

        # Normalize texts
        student_clean = normalize_text(student_answer)
        correct_clean = normalize_text(correct_answer)

        # 1. Semantic Similarity
        emb_student = semantic_model.encode(student_clean, convert_to_tensor=True)
        emb_correct = semantic_model.encode(correct_clean, convert_to_tensor=True)
        similarity = float(util.cos_sim(emb_student, emb_correct).item())
        similarity_score = calculate_score(similarity)

        # 2. NLI Entailment Check
        nli_probs = nli_entailment(correct_clean, student_clean)
        entailment_prob = nli_probs["entailment"]
        nli_score = round(entailment_prob * 10, 2)

        # 3. Keyword Matching
        keyword_score = round(keyword_match_score(correct_clean, student_clean), 2)

        # 4. Numeric Check
        numeric_score = round(numeric_check_score(correct_clean, student_clean), 2)

        # Weights for each component
        w_sim = 0.4
        w_nli = 0.4
        w_keyword = 0.1
        w_numeric = 0.1

        # Combine scores with weights to get a final score
        final_score = (w_sim * similarity_score + 
                       w_nli * nli_score + 
                       w_keyword * keyword_score + 
                       w_numeric * numeric_score)
        final_score = round(final_score, 2)

        # Ensure similarity is bounded [0, 1]
        similarity = round(max(0.0, min(1.0, similarity)), 4)

        return jsonify({
            'score': final_score,      # Final weighted score between 0 and 10
            'similarity': similarity,  # Raw semantic similarity for reference
            'nli_entailment': nli_probs["entailment"],
            'keywordScore': keyword_score,
            'numericScore': numeric_score
        })

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return jsonify({'error': str(e), 'score': 0, 'similarity': 0}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

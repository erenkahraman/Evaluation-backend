from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

# Initialize the model (this will be done once when the server starts)
model = SentenceTransformer('all-MiniLM-L6-v2')

def normalize_text(text):
    # Basic text normalization
    text = text.lower().strip()
    # Remove multiple spaces
    text = ' '.join(text.split())
    return text

def calculate_score(similarity):
    """Convert similarity to a score between 0 and 10"""
    # Ensure similarity is between 0 and 1
    similarity = max(0.0, min(1.0, similarity))
    
    # Apply a more lenient scaling to the similarity score
    # This will give higher scores for reasonable answers while still maintaining the 0-10 range
    if similarity >= 0.7:  # High similarity
        score = 7.0 + (similarity - 0.7) * (3.0 / 0.3)  # Scale 0.7-1.0 to 7-10
    elif similarity >= 0.4:  # Medium similarity
        score = 4.0 + (similarity - 0.4) * (3.0 / 0.3)  # Scale 0.4-0.7 to 4-7
    else:  # Low similarity
        score = similarity * (4.0 / 0.4)  # Scale 0-0.4 to 0-4
    
    # Final clamp and rounding
    score = round(max(0.0, min(10.0, score)), 2)
    return score

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
        
        # Get embeddings
        emb1 = model.encode(student_clean, convert_to_tensor=True)
        emb2 = model.encode(correct_clean, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = float(util.cos_sim(emb1, emb2).item())
        
        # Convert similarity to score (0-10)
        score = calculate_score(similarity)
        
        # Ensure similarity is also properly bounded
        similarity = round(max(0.0, min(1.0, similarity)), 4)

        return jsonify({
            'score': score,  # Will be between 0 and 10
            'similarity': similarity  # Will be between 0 and 1
        })

    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return jsonify({'error': str(e), 'score': 0, 'similarity': 0}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 
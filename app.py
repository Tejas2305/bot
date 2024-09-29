from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Define a route to handle question answering
@app.route('/answer', methods=['POST'])
def answer_question():
    data = request.json
    context = data.get('context')
    question = data.get('question')

    if not context or not question:
        return jsonify({'error': 'Context and question are required.'}), 400

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors='pt', truncation=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end]))

    return jsonify({'answer': answer})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

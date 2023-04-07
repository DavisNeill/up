from flask import Flask, render_template, request
import numpy as np
import torch
from transformers import AlbertTokenizer, AlbertModel

app = Flask(__name__)

questions = [
    {
        "question": "The sun _____ in the east.",
        "choices": ["raise", "rises", "rose", "rising"],
        "correct": 1
    },
    {
        "question": "She has _____ apples.",
        "choices": ["much", "a few", "any", "many"],
        "correct": 3
    },
    {
        "question": "He _____ to the store.",
        "choices": ["goes", "go", "gone", "went"],
        "correct": 0
    }
]

model_name = "albert-base-v2"
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertModel.from_pretrained(model_name)

def generate_embeddings(response):
    inputs = tokenizer(response, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embeddings

question_embeddings = []
for question_data in questions:
    question_text = question_data["question"]
    choices = question_data["choices"]
    choice_embeddings = []
    for choice in choices:
        full_text = question_text.replace("_____", choice)
        embeddings = generate_embeddings(full_text)
        choice_embeddings.append(embeddings)
    question_embeddings.append(np.array(choice_embeddings))

def irt_based_score(response_embeddings, choice_embeddings, correct_choice):
    scores = np.dot(choice_embeddings, response_embeddings)
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)
    return probabilities[correct_choice]

@app.route('/')
def index():
    return render_template('quiz.html', questions=questions)

@app.route('/submit', methods=['POST'])
def submit():
    student_responses = []
    for question_index in range(len(questions)):
        response_index = int(request.form[str(question_index)])
        student_responses.append(response_index)

    student_score = 0
    for question_index, response_index in enumerate(student_responses):
        question_data = questions[question_index]
        correct_choice = question_data["correct"]
        choice_embeddings = question_embeddings[question_index]
        response_embeddings = choice_embeddings[response_index]

        score = irt_based_score(response_embeddings, choice_embeddings, correct_choice)
        student_score += score

    return render_template('score.html', student_score=round(student_score*100,2))

if __name__ == '__main__':
    app.run(debug=True)

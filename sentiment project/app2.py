import os
import re
import pandas as pd
import joblib
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'tsv'}

# Load the model and vectorizer
vectorizer = joblib.load('cvtransform.pkl')
model = joblib.load('model.pkl')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
stemmer = PorterStemmer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = text.split()
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    # Rejoin tokens
    return ' '.join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '' or not allowed_file(file.filename):
                return render_template_string('''
                <!doctype html>
                <title>Invalid File</title>
                <h1>Invalid file format. Please upload a TSV file.</h1>
                <a href="/">Go Back</a>
                ''')

            filename = 'uploaded_file.tsv'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the uploaded file
            try:
                data = pd.read_csv(file_path, delimiter='\t')
                if 'Review' not in data.columns:
                    return render_template_string('''
                    <!doctype html>
                    <title>Missing Column</title>
                    <h1>Uploaded file must contain a 'Review' column.</h1>
                    <a href="/">Go Back</a>
                    ''')

                # Preprocess reviews
                data['cleaned_review'] = data['Review'].apply(preprocess_text)

                corpus = data['cleaned_review'].tolist()
                X_new = vectorizer.transform(corpus).toarray()

                predictions = model.predict(X_new)
                data['predicted_label'] = predictions
                data['predicted_label'] = data['predicted_label'].replace({0: '😔', 1: '😊'})

                # Count sentiments
                sentiment_counts = data['predicted_label'].value_counts()
                total_negative = sentiment_counts.get('😔', 0)
                total_positive = sentiment_counts.get('😊', 0)

                # Convert to HTML
                table_html = data.to_html(classes='table table-striped table-bordered', index=False)

                return render_template_string('''
                <!doctype html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Restaurant Reviews Output</title>
                    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
                    <style>
                        body {
                            padding-top: 20px;
                            background-color: #f8f9fa;
                        }
                        .container {
                            max-width: 900px;
                            margin: 0 auto;
                        }
                        h1 {
                            margin-bottom: 20px;
                        }
                        .table {
                            margin-top: 20px;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Restaurant Reviews Output</h1>
                        <div class="mt-3">
                            {{ table_html | safe }}
                        </div>
                        <div class="mt-3">
                            <h3>Total Positive Sentiments: {{ total_positive }}</h3>
                            <h3>Total Negative Sentiments: {{ total_negative }}</h3>
                        </div>
                        <a class="btn btn-primary mt-3" href="/">Go Back</a>
                    </div>
                </body>
                </html>
                ''', table_html=table_html, total_positive=total_positive, total_negative=total_negative)
            except Exception as e:
                return render_template_string(f'''
                <!doctype html>
                <title>Error</title>
                <h1>An error occurred while processing the file: {e}</h1>
                <a href="/">Go Back</a>
                ''')

        elif 'sentence' in request.form:
            sentence = request.form['sentence']
            try:
                cleaned_sentence = preprocess_text(sentence)
                corpus = [cleaned_sentence]
                X_new = vectorizer.transform(corpus).toarray()

                prediction = model.predict(X_new)[0]

                sentiment = '😊' if prediction == 1 else '😔'

                return render_template_string('''
                <!doctype html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Sentence Sentiment Analysis</title>
                    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
                    <style>
                        body {
                            padding-top: 20px;
                            background-color: #f8f9fa;
                        }
                        .container {
                            max-width: 900px;
                            margin: 0 auto;
                        }
                        h1 {
                            margin-bottom: 20px;
                        }
                        h3 {
                            margin-top: 20px;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Sentence Sentiment Analysis</h1>
                        <div class="mt-3">
                            <h3>Sentiment: {{ sentiment }}</h3>
                        </div>
                        <a class="btn btn-primary mt-3" href="/">Go Back</a>
                    </div>
                </body>
                </html>
                ''', sentiment=sentiment)
            except Exception as e:
                return render_template_string(f'''
                <!doctype html>
                <title>Error</title>
                <h1>An error occurred while processing the sentence: {e}</h1>
                <a href="/">Go Back</a>
                ''')

    return '''
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sentiment Analysis</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <style>
            body {
                padding-top: 20px;
                background-color: #f8f9fa;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
            }
            h1 {
                margin-bottom: 20px;
            }
            form {
                margin-bottom: 20px;
            }
            .btn {
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sentiment Analysis</h1>
            <h2>Analyze a Sentence</h2>
            <form action="" method="post">
                <div class="form-group">
                    <input type="text" name="sentence" class="form-control" placeholder="Enter a sentence" required>
                </div>
                <button type="submit" class="btn btn-primary">Analyze</button>
            </form>
            <h2>Upload a TSV File</h2>
            <form action="" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <input type="file" name="file" class="form-control-file" required>
                </div>
                <button type="submit" class="btn btn-primary">Upload</button>
            </form>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

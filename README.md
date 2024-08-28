# Sentimenter

Sentimenter is a web application for performing sentiment analysis on restaurant reviews. Users can analyze individual sentences or upload TSV files containing multiple reviews. The application uses a pre-trained machine learning model to classify sentiments as positive or negative, represented with happy 😊 and sad 😔 faces respectively.

## Features

- **Sentence Sentiment Analysis**: Input a single sentence to receive its sentiment analysis result.
- **TSV File Upload**: Upload a TSV file containing reviews to receive sentiment predictions for all reviews.
- **Interactive Web Interface**: User-friendly web interface with Bootstrap styling.

## Project Structure

- `app.py`: Main Flask application file containing routes and logic for processing inputs.
- `uploads/`: Directory where uploaded TSV files are stored temporarily.
- `cvtransform.pkl`: Pickled CountVectorizer object used for text feature extraction.
- `model.pkl`: Pickled machine learning model used for sentiment classification.
- `templates/`: Contains HTML templates for rendering the web pages.

## Usage

- **Analyze a Sentence**: Enter a sentence in the provided text input and click "Analyze" to see the sentiment.
  
  ![Sentence Analysis](path/to/sentence_analysis_image.png)  <!-- Add your image here -->

- **Upload a TSV File**: Choose a TSV file containing reviews and click "Upload" to process and view the sentiment analysis results.

  ![TSV Upload](path/to/tsv_upload_image.png)  <!-- Add your image here -->

- **Main Webpage**: The main webpage provides options for analyzing sentences or uploading TSV files.

  ![Main Webpage](path/to/main_webpage_image.png)  <!-- Add your image here -->
  <img width="1440" alt="Screenshot 2024-08-28 at 7 15 07 PM" src="https://github.com/user-attachments/assets/d2ee5edd-2275-4315-abc9-0a34bfd84249">
<img width="1440" alt="Screenshot 2024-08-28 at 7 15 07 PM" src="https://github.com/user-attachments/assets/eb8e3549-ce59-4406-b8e5-067c627cc2c6">


## Dependencies

- Flask
- Pandas
- scikit-learn
- NLTK
- joblib

You can find the full list of dependencies in `requirements.txt`.

## Troubleshooting

- **File Upload Issues**: Ensure the uploaded file is in TSV format and contains a 'Review' column.
- **Model Errors**: Ensure that `cvtransform.pkl` and `model.pkl` are correctly placed and compatible with the code.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## Contact

For any questions or feedback, please reach out to [Vipul22576@iiitd.ac.in](mailto:Vipul22576@iiitd.ac.in).

---

Replace `path/to/sentence_analysis_image.png`, `path/to/tsv_upload_image.png`, and `path/to/main_webpage_image.png` with the actual paths to your images.

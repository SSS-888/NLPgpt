from flask import Flask, render_template, request, flash
import logging
import nlp_utils

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        case_text = request.form.get('case')
        top_n = int(request.form.get('top_n', 5))  # Get the value of top_n from the form, default to 5

        if not file or not case_text:
            flash("Please upload a file and enter a case text.", "error")
            return render_template('index.html')

        try:
            file_content = file.read()
            sentences = nlp_utils.load_and_preprocess(file_content)
            relevant_sentences = nlp_utils.extract_relevant_sentences(sentences, case_text, top_n)
            return render_template('result.html', case=case_text, relevant_sentences=relevant_sentences, top_n=top_n)
        except UnicodeDecodeError:
            flash("Error decoding the text file. Please ensure it's in a valid format.", "error")
        except ValueError:
            flash("Top N value must be a positive integer.", "error")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            flash("An error occurred during processing. Please try again.", "error")

    return render_template('index.html')

@app.context_processor
def inject_instructions():
    instructions = """
    To achieve better accuracy, follow these guidelines for creating the text file:

    1. Use a plain text file format (.txt) without any formatting or special characters.
    2. Ensure the text is well-structured, with proper sentence and paragraph separation.
    3. Avoid excessive abbreviations or acronyms, as they may reduce accuracy.
    4. Use clear and concise language, avoiding ambiguity or complex phrasing.
    5. Include relevant context and background information related to the case or topic.
    6. The more detailed and explanatory the text, the better the accuracy will be.

    Follow these instructions to get the most relevant and accurate results from the analysis.
    """
    return dict(instructions=instructions)

if __name__ == '__main__':
    app.run(debug=True)

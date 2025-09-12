# PDF OCR Processor

A command-line tool to extract text from PDFs, perform OCR, and run
basic NLP analysis.

------------------------------------------------------------------------
## 📌 Usage Examples with WEB Streamlit UI

Save the code as **`pdf_ocr_processor_with_streamlite_ui.py`** and run:
### Basic usage

``` bash
streamlit run pdf_ocr_processor_with_streamlite_ui.py 
```


## 📌 Usage Examples with CMD line prompt

Save the code as **`pdf_ocr_processor.py`** and run:

### Basic usage

``` bash
python pdf_ocr_processor.py document.pdf
```

### Specify output file

``` bash
python pdf_ocr_processor.py document.pdf --output results.txt
```

### Higher DPI for better quality (slower)

``` bash
python pdf_ocr_processor.py document.pdf --dpi 400
```

### Skip NLP analysis

``` bash
python pdf_ocr_processor.py document.pdf --no-nlp
```

### Verbose output

``` bash
python pdf_ocr_processor.py document.pdf --verbose
```

### Help message

``` bash
python pdf_ocr_processor.py --help
```

------------------------------------------------------------------------

## 📂 Output File Format

The output text file will contain:

-   Extracted text from all pages\
-   Named entities found in the text\
-   Frequency counts of nouns and verbs\
-   Clear section headers and formatting

------------------------------------------------------------------------

## ⚙️ Requirements

Make sure the following libraries are installed:

``` bash
pip install streamlit pytesseract pdf2image opencv-python pillow pymupdf spacy pandas numpy
```

Install the spaCy model:

``` bash
python -m spacy download en_core_web_sm
```

------------------------------------------------------------------------

✅ This command-line version maintains all the functionality of your
original **Streamlit app** but provides a professional interface
suitable for **scripting and batch processing**.

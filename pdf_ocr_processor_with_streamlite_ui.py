import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
import cv2
import numpy as np
from PIL import Image
import io
import spacy
from spacy import displacy
import pandas as pd
import tempfile
import os
import fitz  # PyMuPDF
import base64

# Set page configuration
st.set_page_config(
    page_title="Scanned PDF OCR & NLP Processor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.error("spaCy model 'en_core_web_sm' not found. Please install it with: python -m spacy download en_core_web_sm")
        return None

nlp = load_spacy_model()

# Function to enhance image for better OCR
def enhance_image(image):
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply dilation to make text more visible
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return dilated

# Function to perform OCR on an image
def ocr_from_image(image):
    enhanced = enhance_image(image)
    text = pytesseract.image_to_string(enhanced, config='--psm 6')
    return text

# Alternative PDF processing using PyMuPDF if pdf2image fails
def process_pdf_with_pymupdf(uploaded_file):
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())  # Use getvalue() instead of read()
        tmp_path = tmp_file.name
    
    try:
        doc = fitz.open(tmp_path)
        extracted_text = ""
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num in range(len(doc)):
            status_text.text(f"Processing page {page_num+1}/{len(doc)}...")
            progress_bar.progress((page_num + 1) / len(doc))
            
            page = doc.load_page(page_num)
            # Try to extract text directly first
            text = page.get_text()
            if text.strip():  # If text extraction worked
                extracted_text += f"--- Page {page_num+1} ---\n\n{text}\n\n"
            else:
                # If no text found, it's likely a scanned document - convert to image for OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                # Convert PIL image to OpenCV format
                open_cv_image = np.array(image.convert('RGB'))
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
                
                # Perform OCR
                page_text = ocr_from_image(open_cv_image)
                extracted_text += f"--- Page {page_num+1} ---\n\n{page_text}\n\n"
        
        status_text.text("OCR completed!")
        progress_bar.empty()
        status_text.empty()
        
        return extracted_text
    except Exception as e:
        st.error(f"Error processing PDF with PyMuPDF: {str(e)}")
        return ""
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

# Function to process PDF and extract text
def process_pdf(uploaded_file):
    try:
        # First try using pdf2image
        file_bytes = uploaded_file.getvalue()  # Get file bytes once
        images = convert_from_bytes(file_bytes, dpi=300)
        
        extracted_text = ""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image in enumerate(images):
            status_text.text(f"Processing page {i+1}/{len(images)}...")
            progress_bar.progress((i + 1) / len(images))
            
            # Convert PIL image to OpenCV format
            open_cv_image = np.array(image.convert('RGB'))
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
            
            # Perform OCR
            page_text = ocr_from_image(open_cv_image)
            extracted_text += f"--- Page {i+1} ---\n\n{page_text}\n\n"
        
        status_text.text("OCR completed!")
        progress_bar.empty()
        status_text.empty()
        
        return extracted_text
    
    except Exception as e:
        st.warning(f"pdf2image failed: {str(e)}. Falling back to PyMuPDF method...")
        # Fall back to PyMuPDF method
        return process_pdf_with_pymupdf(uploaded_file)

# Function to analyze text with spaCy
def analyze_text(text):
    if nlp is None:
        return None, None, None
    
    # Limit text length for performance
    text = text[:100000] if len(text) > 100000 else text
    
    doc = nlp(text)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extract nouns and verbs
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    
    return entities, nouns, verbs

# Function to check if Tesseract is available
def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
        return True
    except:
        return False

# Main application
def main():
    st.title("📄 Scanned PDF OCR & NLP Processor")
    st.markdown("""
    This tool extracts text from scanned PDFs using OCR and then applies Natural Language Processing 
    to identify entities, nouns, and verbs in the text.
    """)
    
    # Check if Tesseract is available
    if not check_tesseract():
        st.error("""
        Tesseract OCR is not installed or not found in PATH. Please install it:
        
        **Windows:** Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
        
        **macOS:** `brew install tesseract`
        
        **Linux:** `sudo apt install tesseract-ocr`
        """)
        return
    
    # Installation instructions in sidebar
    st.sidebar.header("Installation Requirements")
    st.sidebar.markdown("""
    **For optimal performance:**
    
    **Windows:**
    - Download Poppler: [Link](https://github.com/oschwartz10612/poppler-windows/releases/)
    - Add to PATH or set poppler_path in code
    
    **macOS:**
    ```bash
    brew install poppler tesseract
    ```
    
    **Linux:**
    ```bash
    sudo apt install poppler-utils tesseract-ocr
    ```
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload a scanned PDF", type="pdf")
    
    if uploaded_file is not None:
        # Display file info
        file_size = len(uploaded_file.getvalue()) / 1024  # Size in KB
        st.sidebar.info(f"File: {uploaded_file.name}\nSize: {file_size:.2f} KB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("OCR Results")
            with st.spinner("Processing PDF..."):
                extracted_text = process_pdf(uploaded_file)
            
            if extracted_text:
                st.text_area("Extracted Text", extracted_text, height=300)
                
                # Download option for extracted text
                st.download_button(
                    label="Download Extracted Text",
                    data=extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
            else:
                st.error("Failed to extract text from the PDF.")
        
        with col2:
            st.subheader("NLP Analysis")
            if extracted_text and nlp:
                with st.spinner("Analyzing text..."):
                    entities, nouns, verbs = analyze_text(extracted_text)
                
                if entities:
                    st.write("**Named Entities Found:**")
                    entity_df = pd.DataFrame(entities, columns=["Entity", "Type"])
                    st.dataframe(entity_df, use_container_width=True)
                else:
                    st.info("No named entities found in the text.")
                
                if nouns:
                    st.write("**Frequent Nouns:**")
                    noun_series = pd.Series(nouns).value_counts().head(10)
                    st.bar_chart(noun_series)
                
                if verbs:
                    st.write("**Frequent Verbs:**")
                    verb_series = pd.Series(verbs).value_counts().head(10)
                    st.bar_chart(verb_series)
            else:
                st.info("No text available for NLP analysis.")
    
    else:
        st.info("Please upload a scanned PDF file to begin processing.")

if __name__ == "__main__":
    main()
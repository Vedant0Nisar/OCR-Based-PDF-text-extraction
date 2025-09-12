import argparse
import pytesseract
from pdf2image import convert_from_bytes
import cv2
import numpy as np
from PIL import Image
import io
import spacy
import pandas as pd
import tempfile
import os
import fitz  # PyMuPDF
import sys
from pathlib import Path

def setup_arg_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="PDF OCR and NLP Processor - Extract text from scanned PDFs and perform NLP analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_ocr_processor.py document.pdf
  python pdf_ocr_processor.py document.pdf --output results.txt
  python pdf_ocr_processor.py document.pdf --no-nlp --dpi 400
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the PDF file to process"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: input_filename_processed.txt)",
        default=None
    )
    
    parser.add_argument(
        "--dpi",
        help="DPI for image conversion (default: 300)",
        type=int,
        default=300
    )
    
    parser.add_argument(
        "--no-nlp",
        help="Skip NLP analysis",
        action="store_true"
    )
    
    parser.add_argument(
        "--verbose",
        help="Enable verbose output",
        action="store_true"
    )
    
    return parser

def log_message(message, verbose=False):
    """Print message if verbose mode is enabled"""
    if verbose:
        print(f"[INFO] {message}")

def enhance_image(image):
    """Enhance image for better OCR results"""
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

def ocr_from_image(image):
    """Perform OCR on an image"""
    enhanced = enhance_image(image)
    text = pytesseract.image_to_string(enhanced, config='--psm 6')
    return text

def process_pdf_with_pymupdf(file_path, dpi=300, verbose=False):
    """Process PDF using PyMuPDF (fallback method)"""
    try:
        doc = fitz.open(file_path)
        extracted_text = ""
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            log_message(f"Processing page {page_num+1}/{total_pages} with PyMuPDF", verbose)
            
            page = doc.load_page(page_num)
            # Try to extract text directly first
            text = page.get_text()
            if text.strip():  # If text extraction worked
                extracted_text += f"--- Page {page_num+1} ---\n\n{text}\n\n"
            else:
                # If no text found, convert to image for OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                
                # Convert PIL image to OpenCV format
                open_cv_image = np.array(image.convert('RGB'))
                open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
                
                # Perform OCR
                page_text = ocr_from_image(open_cv_image)
                extracted_text += f"--- Page {page_num+1} ---\n\n{page_text}\n\n"
        
        return extracted_text
    except Exception as e:
        print(f"[ERROR] Failed to process PDF with PyMuPDF: {str(e)}")
        return ""

def process_pdf(file_path, dpi=300, verbose=False):
    """Main function to process PDF and extract text"""
    try:
        # First try using pdf2image
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        images = convert_from_bytes(file_bytes, dpi=dpi)
        extracted_text = ""
        total_pages = len(images)
        
        for i, image in enumerate(images):
            log_message(f"Processing page {i+1}/{total_pages} with pdf2image", verbose)
            
            # Convert PIL image to OpenCV format
            open_cv_image = np.array(image.convert('RGB'))
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
            
            # Perform OCR
            page_text = ocr_from_image(open_cv_image)
            extracted_text += f"--- Page {i+1} ---\n\n{page_text}\n\n"
        
        return extracted_text
    
    except Exception as e:
        log_message(f"pdf2image failed: {str(e)}. Falling back to PyMuPDF method...", verbose)
        # Fall back to PyMuPDF method
        return process_pdf_with_pymupdf(file_path, dpi, verbose)

def analyze_text(text, verbose=False):
    """Analyze text with spaCy NLP"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        print("[WARNING] spaCy model 'en_core_web_sm' not found. NLP analysis skipped.")
        return None, None, None
    
    # Limit text length for performance
    text = text[:100000] if len(text) > 100000 else text
    
    log_message("Performing NLP analysis...", verbose)
    doc = nlp(text)
    
    # Extract entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Extract nouns and verbs
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]
    
    return entities, nouns, verbs

def save_results(extracted_text, entities, nouns, verbs, output_path, verbose=False):
    """Save results to output file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PDF OCR EXTRACTION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("EXTRACTED TEXT:\n")
            f.write("=" * 80 + "\n")
            f.write(extracted_text)
            
        
        log_message(f"Results saved to: {output_path}", verbose)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save results: {str(e)}")
        return False

def main():
    """Main function"""
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {args.input_file}")
        sys.exit(1)
    
    if input_path.suffix.lower() != '.pdf':
        print(f"[ERROR] Input file must be a PDF: {args.input_file}")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_processed.txt"
    
    print(f"Processing PDF: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Process the PDF
    extracted_text = process_pdf(str(input_path), args.dpi, args.verbose)
    
    if not extracted_text:
        print("[ERROR] Failed to extract text from PDF")
        sys.exit(1)
    
    print(f"Successfully extracted {len(extracted_text)} characters of text")
    
    # Perform NLP analysis if not disabled
    entities, nouns, verbs = None, None, None
    if not args.no_nlp:
        entities, nouns, verbs = analyze_text(extracted_text, args.verbose)
    
    # Save results
    if save_results(extracted_text, entities, nouns, verbs, str(output_path), args.verbose):
        print(f"Processing complete! Results saved to: {output_path}")
    else:
        print("[ERROR] Failed to save results")
        sys.exit(1)

if __name__ == "__main__":
    main()
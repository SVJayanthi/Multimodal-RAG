from pathlib import Path
from pdf_loaders.image_loader import load_pdf_image_content
from pdf_loaders.text_loader import load_pdf_text_content
import json
import os

def chunk_and_save_docs(docs_dir, save_images_dir, jsonl_path):
    # file_to_page_to_image, image_dicts = load_pdf_image_content(docs_dir, save_images_dir=save_images_dir)
    
    text_dicts = load_pdf_text_content(docs_dir)
    # set image paths for parents
    for p_dict in text_dicts:
        p_metadata = p_dict['metadata']
        pdf_stem = Path(p_metadata['filename']).stem
        page_num = p_metadata['page_number']
        page_num_file = "page"+str(page_num)+".jpg"
        
        # if pdf_stem in file_to_page_to_image and page_num in file_to_page_to_image[pdf_stem]:
        #     image_path = os.path.join(save_images_dir, pdf_stem, page_num_file)
        #     p_metadata['image_location'] = image_path
        # else:
        #     p_metadata['image_location'] = ""
        
    chunked_corpus = text_dicts + image_dicts
    
    # Open a file in write mode
    with open(jsonl_path, 'w') as f:
        for obj in chunked_corpus:
            json_str = json.dumps(obj)
            f.write(json_str + '\n')
            
chunk_and_save_docs("C:/Users/srava/Documents/Interview Coding + Design/gleen/docs", None, None)
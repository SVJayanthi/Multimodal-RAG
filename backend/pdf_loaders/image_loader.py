from pathlib import Path
import layoutparser as lp
import cv2
from pdf2image import convert_from_path
from PIL import Image
import os
from utils.utils import generate_random_id
import datetime
from verbalizers.openai_verbalizer import OpenAIVerbalizer

def parse_pdf_images(docs_dir, save_dir):
  file_to_page_to_image = {}
  generated_image_paths = []
  for file in os.listdir(docs_dir):
      file_path = os.path.join(docs_dir, file)
      pdf_stem = Path(file_path).stem
      file_to_page_to_image[pdf_stem] = {}
      
      # Store Pdf with convert_from_path function
      images = convert_from_path(file_path)
      save_images_pdf_dir = os.path.join(save_dir, pdf_stem)
      os.makedirs(save_images_pdf_dir, exist_ok=True)
      for i in range(len(images)):
        page_img_name= 'page'+ str(i) +'.jpg'
        save_path = os.path.join(save_images_pdf_dir, page_img_name)
        # Save pages as images in the pdf
        images[i].save(save_path, 'JPEG')
        generated_image_paths.append(save_path)
        file_to_page_to_image[pdf_stem][i] = save_path
  return file_to_page_to_image, generated_image_paths

# Save cropped image
def save_crop_images(input_path, output_dir, figure_blocks):
    saved_locations = []
    counter = 0
    for figure in figure_blocks:
        bounding_box = figure.block
        coords = [bounding_box.x_1, bounding_box.y_1, bounding_box.x_2, bounding_box.y_2]

        output_path = os.path.join(output_dir, f"fig_{counter}.jpg")
        # Open the image
        with Image.open(input_path) as img:
            # Crop the image
            cropped_image = img.crop(coords)

            os.makedirs(Path(output_path).parent, exist_ok=True)
            # Save the cropped image
            cropped_image.save(output_path)
            saved_locations.append(output_path)
    return saved_locations

def get_figure_blocks(model, image_paths, cropped_img_dir = "cropped_images/"):
    list_of_fig_blocks = []
    cropped_image_paths = []
    for image_path in image_paths:
        path_obj = Path(image_path)
        parent = path_obj.parent
        stem = path_obj.stem
        image = cv2.imread(image_path)
        image = image[..., ::-1]
        layout = model.detect(image)
        figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])
        list_of_fig_blocks.extend(figure_blocks)
        
        save_crop_dir = os.path.join(parent, cropped_img_dir, stem)
        cropped_image_paths.extend(save_crop_images(image_path, save_crop_dir, figure_blocks))
    return list_of_fig_blocks, cropped_image_paths

def verbalize_figures(verbalizer, list_of_fig_blocks, cropped_image_paths, pdf_dir):
    image_dicts = []
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%dT%H:%M:%S')
    
    for fig, loc in zip(list_of_fig_blocks, cropped_image_paths):
        page_num = int(Path(loc).parent.stem.replace("page", ""))
        pdf_path = Path(loc).parent.parent.parent.stem + ".pdf"
        
        random_id = generate_random_id()
        image_verbalization = verbalizer(loc)
        # Add to docs dictionary
        image_dicts.append({'type': 'Image',
            'element_id': random_id,
            'text': image_verbalization,
            'metadata': {'coordinates': {'points': ((fig.block.x_1,
                fig.block.y_2),
                (fig.block.x_1, fig.block.y_1),
                (fig.block.x_2, fig.block.y_1),
                (fig.block.x_2, fig.block.y_2)),
                },
            'image_location': loc,
            'file_directory':  pdf_dir,
            'filename': pdf_path,
            'languages': ['eng'],
            'last_modified': formatted_datetime,
            'page_number': page_num,
            'filetype': 'application/pdf'}
        })
    return image_dicts

def load_pdf_image_content(docs_dir, save_images_dir):
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    
    verbalizer = OpenAIVerbalizer()
    
    file_to_page_to_image, pdf_image_paths = parse_pdf_images(docs_dir, save_images_dir)
    list_of_fig_blocks, cropped_image_paths = get_figure_blocks(model, pdf_image_paths)
    image_dicts = verbalize_figures(verbalizer, list_of_fig_blocks, cropped_image_paths, docs_dir)

    return file_to_page_to_image, image_dicts
    
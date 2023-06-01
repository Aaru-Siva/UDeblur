import streamlit as st
import os
import shutil

from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import time

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def single_image_inference(model, img, save_path):
    start_time = time.time()  # Start timing
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, save_path)

    end_time = time.time()  # End timing
    duration = end_time - start_time  # Calculate duration
    return duration

def main():
    # Create a folder to temporary store uploaded files
    upload_folder = 'upload/input'
    result_folder = 'upload/output'

    if os.path.isdir(upload_folder):
        shutil.rmtree(upload_folder)
    if os.path.isdir(result_folder):
        shutil.rmtree(result_folder)
    os.makedirs(upload_folder)
    os.makedirs(result_folder)

    # Load the model
    opt_path = 'options/test/GoPro/NAFNet-width64.yml' 
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)

    # Create the Streamlit app
    st.title('UDeblur - Image Deblurring App')

    # Create a list of items
    items = ["Upload blur image below", "You can upload images in PNG, JPG, JPEG, BMP and WEBP format", "You can able to deblur multiple images one after another without going back"]

    # Use the st.write() function and Markdown syntax to create bullet points
    st.write("- " + "\n- ".join(items))

    # Add a file uploader to the app
    uploaded_files = st.file_uploader('Upload an image', accept_multiple_files=True)

    if uploaded_files is not None:
      durations = []
      for file in uploaded_files:
          if not file.name.endswith(('.jpg', '.jpeg', '.png','.webp','.bmp')):
              st.error('Oops! Wrong input type. Please upload image type file.')
          else:
            # Process the uploaded images
            if uploaded_files:
                    for uploaded_file in uploaded_files:
                        # Save the uploaded image to the input folder
                        with open(os.path.join(upload_folder, uploaded_file.name), 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Load the input image and process it with the model
                        input_path = os.path.join(upload_folder, uploaded_file.name)
                        img_input = imread(input_path)
                        inp = img2tensor(img_input)
                        output_path = os.path.join(result_folder, uploaded_file.name)
                        duration = single_image_inference(NAFNet, inp, output_path)
                        durations.append(duration)

                    # Display the input and output images
                    st.success('Congratulations! You have successfully completed the operation.')
                    input_list = sorted(os.listdir(upload_folder))
                    output_list = sorted(os.listdir(result_folder))
                    for input_path, output_path, duration in zip(input_list, output_list, durations):
                        st.write(f"Time taken to deblur the image {input_path} is **{duration:.2f} seconds**")
                        st.image([os.path.join(upload_folder, input_path), os.path.join(result_folder, output_path)],
                                caption=['Blurred Image', 'Deblurred Image'],
                                width=250)

if __name__ == '__main__':
    main()

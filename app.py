"""
A streamlit app which does the following:
- Takes the images path, model names, seed and device as input.
- Performs img2img for every image in the path by choosing a random model from the model names.
- Displays the result in a grid view.
"""

import random
import logging

import torch
import streamlit as st

from diffusion import DiffusionDataset, DiffusionModel

from profile_utils import profile


def get_available_devices() -> list:
    """Gets the available devices for the inference to run.

    Returns:
        list: A list of devices on which the inference can be performed.
    """
    available_devices = []
    if torch.cuda.is_available():
        available_gpus = [f"cuda:{i}-{torch.cuda.get_device_name(i)}"
                          for i in range(torch.cuda.device_count())]
        available_devices.extend(available_gpus)
    available_devices.append("cpu-default")
    return available_devices


@profile
def img2img(image_folder_url:str,
            model_text_file_url:str,
            image_size:tuple,
            seed:int, device:str) -> None:
    """A method which performs img2img with the dataset on a specified device.

    Args:
        image_folder_url (str): The directory containing the images.
        model_text_file_url (str): The text file containing the list of models.
        image_size (tuple): A tuple containing the required width and the height of the image.
        seed (int): The starting seed.
        device (str): The device to be selected for inference.
    """
    hash_value = random.getrandbits(128)
    logging.basicConfig(filename="log.txt",
                    filemode='w',
                    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.INFO)
    logging.info(f"{'-' * 30} RUN STARTED WITH HASH: {hash_value} {'-' * 30}")

    # Load the dataset, device and model.
    dataset = DiffusionDataset(images_dir=image_folder_url, models_text_file=model_text_file_url, seed=seed)
    device_name = device.split("-")[0]
    diff_obj = DiffusionModel(dataset, image_size, device=device_name)
    result_images = []
    result_captions = []

    # Load the UI components for progress bar and image grid
    progress_bar_ui = st.empty()
    with progress_bar_ui.container():
        progress_bar = st.progress(0, text=f"Performing inference on {len(dataset)} images...")
    image_grid_ui = st.empty()

    # Run inference for every image, prompt and model name in the dataset
    for i, (image_path, prompt, model_name) in enumerate(dataset):
        result = diff_obj.img2img_single_image(image_path, prompt, model_name)
        result_images.append(result)
        result_captions.append(prompt)

        # Start with empty UI elements
        progress_bar_ui.empty()
        image_grid_ui.empty()

        # Update the progress bar
        with progress_bar_ui.container():
            value = ((i+1)/(len(dataset)))
            progress_bar.progress(value, text=f"{i+1} out of {len(dataset)} images processed.")

        # Update the image grid
        with image_grid_ui.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                for i in range(0, len(result_images), 3):
                    st.image(result_images[i], caption=result_captions[i])
            with col2:
                for i in range(1, len(result_images), 3):
                    st.image(result_images[i], caption=result_captions[i])
            with col3:
                for i in range(2, len(result_images), 3):
                    st.image(result_images[i], caption=result_captions[i])


if __name__ == "__main__":
    st.title("Img2Img inference HuggingFace")
    with st.form(key='form_parameters'):
        images_dir = st.text_input("Enter the parent folder path")
        model_text_file_path = st.text_input("Enter the model text file path")
        
        width_col1, height_col2 = st.columns(2)
        with width_col1:
            width_input = st.number_input("Image width (default=256)", value=256, min_value=1)
        with height_col2:
            height_input = st.number_input("Image height (default=256)", value=256, min_value=1)

        col1_inp, col2_inp = st.columns(2)
        with col1_inp:
            seed_input = int(st.number_input("Enter the seed (default=25)", value=25, min_value=0))
        with col2_inp:
            device_input = st.selectbox("Select the device", get_available_devices())

        submitted = st.form_submit_button("Predict")

    if submitted: # The form is submitted
        img2img(images_dir, model_text_file_path, (width_input, height_input), seed_input, device_input)
        logging.info(f"{'-' * 30} RUN HAS ENDED {'-' * 30}")

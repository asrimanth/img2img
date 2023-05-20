"""
The diffusion module to perform inference and record metrics for img2img.

"""

import os
import random
import logging

import torch
from torch.utils.data import Dataset

from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline
from profile_utils import profile


class DiffusionDataset(Dataset):
    def __init__(self, images_dir: str, models_text_file:str, seed:int = 25) -> None:
        """A PyTorch dataset for diffusion inference. Loads the image from the images dir.
        Selects a random model from the text file for each image.
        Returns a tuple of image_path, prompt(filename), and a random model name.

        Args:
            images_dir (str): The input path which contains all the images.
            models_text_file (str): A text file containing the model names.
            seed (int, optional): A seed value for the random number generator. Defaults to 25.
        """
        images_dir = images_dir + "/" if images_dir[-1] != "/" else images_dir
        self.image_paths = [images_dir + img_name for img_name in sorted(os.listdir(images_dir))]
        self.image_names = sorted(os.listdir(images_dir))
        with open(models_text_file, encoding="utf-8") as file:
            self.models = [line.rstrip('\n') for line in file]
        self._set_seed(seed)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image_path = self.image_paths[idx]
        prompt_text = " ".join(self.image_names[idx].split("_"))
        model_name = self.models[random.randint(0, len(self.models)-1)]
        return image_path, prompt_text, model_name

    def _set_seed(self, seed:int) -> None:
        """Sets the initial seed of the available modules.

        Args:
            seed (int): Initial seed value for the random number generator.
        """
        random.seed(seed)
        torch.manual_seed(seed)


class DiffusionModel:
    def __init__(self, diffusion_data:DiffusionDataset, image_size:tuple, device:str) -> None:
        """Performs the img2img on the given dataset.

        Args:
            diffusion_data (DiffusionDataset): The dataset containing image, prompt and model name.
            image_size (tuple): The width and height od the image as a tuple.
            device (str): The device for inference.
        """
        self.diffusion_data = diffusion_data
        self.image_size = image_size
        self.device = device
        self.cuda_device_id = None
        if "cuda" in self.device:
            if ":" in self.device:
                self.cuda_device_id = int(self.device.split(":")[-1])
            else:
                self.cuda_device_id = 0
            logging.info(f"Running on device: {torch.cuda.get_device_name(self.cuda_device_id)}")
        else:
            logging.info(f"Running on device: {self.device}")


    @profile
    def img2img_single_image(self, image_path:str, prompt:str, model_name:str) -> Image:
        """Performs img2img on a set of images and models.

        Args:
            image_path (str): The input image path.
            prompt (str): The prompt for img2img.
            model_name (str): The model id as an input for the HuggingFace diffusion pipeline.

        Returns:
            Image: The result image in PIL.Image format.
        """
        init_image = Image.open(image_path).convert("RGB")
        init_image = init_image.resize(self.image_size)
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(model_name,
                                                    cache_dir="/l/vision/v5/sragas/",
                                                    torch_dtype=torch.float32).to(self.device)
        result_images = pipeline(prompt=prompt,
                                 image=init_image,
                                 strength=0.75,
                                 guidance_scale=7.5).images
        if self.cuda_device_id is not None:
            logging.info(f"GPU Mem Allocated: {torch.cuda.memory_allocated(self.cuda_device_id)/1024**3:3.3f} GB")
            logging.info(f"GPU Mem Cached: {torch.cuda.memory_reserved(self.cuda_device_id)/1024**3:3.3f} GB")
        return result_images[0]


if __name__ == "__main__":
    dataset = DiffusionDataset(images_dir="./images/", models_text_file="./model_names.txt")
    diffusion = DiffusionModel(dataset, image_size=(256, 256), device="cpu")

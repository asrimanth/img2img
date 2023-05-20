# Img2Img inference HuggingFace

## Environment setup

This application performs well on CUDA enabled Linux devices. The exact environment can be found in the requirememts_linux_cuda.txt file. It is recommended to create a python virtual environment first and then  installed using the following command:

```{bash}
pip3 install requirememts_linux_cuda.txt
```

For a fresh install on Windows, Linux or Mac, consider using the installation.sh script or the following:
Create and activate a virtual environment. Then run:

```{bash}
pip3 install torch torchvision torchaudio Pillow transformers diffusers["torch"] streamlit
```

## Running the application

After the setup, the app can be run using the following command:

```{bash}
streamlit run app.py
```

Check the terminal and open the url to view the app.

## Design considerations for the application

The code is readable and modular. Doc strings, comments and type hinting is present for every method and module. Pylint was used to check the code standards and the scores were 8+/10.

### The frontend (app.py)

The logic to create a web application around the diffusion is in this file. We take in the image folder path, model text file path, image width, image height, seed, device, and the cache folder path as user input. Once the "Predict" button is clicked, the img2img method is invoked, which calls the backend class for inference. For each image inferred, the image is shown in the grid below. A progress bar shows the number of images inferred.

### The backend (diffusion.py)

The logic to perform inference is written in this file. First, a PyTorch dataset (DiffusionDataset) is created which takes in the path of the images folder and the path of the model_names.txt file.

Assumptions:

+ There is the images folder and the model_names.txt file.
+ The prompt can be found in the file name with underscores separating each word (optional).
+ Image resize is done to ensure faster inference.

Once these are available, the paths of all the images are stored in a list. For each image, a random model from the model_names.txt file is chosen. Once this is done, the image path, prompt (from the file name), and the model name are returned.

The second class is the DiffusionModel class, which contains the logic to perform img2img on a single image. An object oriented approach is used so that more methods can be added in the future if necessary.

The img2img is performed by taking each item in the dataset (image path, prompt, model name) and using the HuggingFace diffusers library. The images are loaded using Pillow just before inference to reduce memory consumption.

### Logging the Metrics

The metrics are tracked using the @profile decorator, available in the profile_utils.py file. The logging and psutil libraries are used to log the time and memory usage. If cuda is available, we also log the GPU memory usage. The logs are stored in the log.txt file. Every time the "Predict" button is clicked, a hash is created for the run and the metrics are tracked.

## Points to improve

+ Tracking of metrics is a bit tricky.
+ The GPU used in the demo is very old. Can use a much better GPU for faster inference.
+ I tried to create a docker image, but the university server did not have docker installed.
+ This application scaled with a powerful GPU on a Kubernetes cluster can perform well as an MVP.

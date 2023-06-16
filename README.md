# DeepFloyd IF Lab

Advanced notebook-based web UI for [DeepFloyd IF](https://github.com/deep-floyd/IF)

## Features

* One-click installation on Windows and Linux.
* Support of all four default IF pipelines: 
    * Dream
    * Style Transfer
    * Super Resolution
    * Inpainting
* PNGInfo
* Convenient batch generation workflow.
* Full control of IF stage parameters.
* JupyterLab environment.
* IF scripting directly in the notebook.

## Minimum System Requirements

* 16GB of system RAM (32GB or more is recommended).
* 12GB of VRAM (24GB is recommended).
* 50GB of disk space (SSD is recommended).

## Installation 

### Obtaining the Huggingface token

1. Make sure to have a [Hugging Face account](https://huggingface.co/join) and be logged in.
2. Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0).
3. Copy your Huggingface [token](https://huggingface.co/settings/tokens) to the clipboard to make it available for the
   installer.

### Windows 10/11

Download the [install.cmd](https://raw.githubusercontent.com/GChristensen/deepfloyd_if_lab/main/scripts/install.cmd) 
script and place it in the directory where you want DeepFloyd IF Lab to be installed.
Launch the script and wait until the browser window with the notebook opens and downloads all the necessary checkpoints.
The installation may take around an hour with an average internet connection and should look as shown in 
[this video](https://youtu.be/0O12LeWgVQY).

### Linux

1. Install [Git](https://git-scm.com/download/linux).
2. Make sure that Python 3.10.9 is the default Python implementation in your system. Only Python 3.10 is supported.
3. Download the [install.sh](https://raw.githubusercontent.com/GChristensen/deepfloyd_if_lab/main/scripts/install.sh)
   script and place it in the directory where you want DeepFloyd IF Lab to be installed.
   Launch the script and wait until the browser window with the notebook opens. It may be necessary to open it manually
   at http://localhost:18888/lab and manually launch the first notebook cell if it does not run automatically.

### Mac

Download the [install.sh](https://raw.githubusercontent.com/GChristensen/deepfloyd_if_lab/main/scripts/install.sh)
script and run it in the directory where you want DeepFloyd IF Lab to be installed. The script will also 
automatically install brew, python, and git.


## Peak VRAM Usage

Peak DeepFloyd IF Lab GPU memory usage for different sets of models and memory layouts, ±1GB.

| Model set                 | Stage produced | I+II+III (24GB) | I/II+III (12GB)   | I/II/III (8GB) |
|---------------------------|----------------|-----------------|-------------------|----------------|
| IF-I-XL + IF-II-L         | Stage II       | 16GB            | NA                |  NA
| IF-I-XL + IF-II-L + SDx4  | Stage III      | 22GB            | 12GB              |  12GB
| IF-I-L + IF-II-L          | Stage II       | 9GB             | NA                |  NA
| IF-I-L + IF-II-L + SDx4   | Stage III      | 12GB            | 12GB              |  8GB
| IF-I-L + IF-II-M          | Stage II       | 7GB             | NA                |  NA
| IF-I-L + IF-II-M + SDx4   | Stage III      | 12GB            | 11GB              |  7GB
| IF-I-M + IF-II-M          | Stage II       | 7GB             | NA                |  NA 
| IF-I-M + IF-II-M + SDx4   | Stage III      | 12GB            | 11GB              |  7GB

The UI attempts to apply the optimal settings depending on the available amount of VRAM. 
Please see [this](https://github.com/GChristensen/deepfloyd_if_lab/wiki/Memory-Management) wiki page for more details.

## Screenshots
| Dream | Style Transfer |
|---|---|
|<a target="_blank" href="https://user-images.githubusercontent.com/170405/243439368-5aeefed6-594b-430a-9414-fb8abfaf5c76.png"><img align="left" src="https://user-images.githubusercontent.com/170405/243440233-8a13ad68-4656-4e40-8e2e-43f36cc31b0d.png"></a>|<a target="_blank" href="https://user-images.githubusercontent.com/170405/243439363-d8c91795-5d22-4b20-a5b3-185e932166b8.png"><img src="https://user-images.githubusercontent.com/170405/243440231-2968d367-81bb-47fe-a223-73ba51eb9ffc.png"></a>|

| Inpainting | Super Resolution |
|------------|------------------|
|<a target="_blank" align="left" href="https://user-images.githubusercontent.com/170405/243439342-8ac57d06-aa46-4214-b4eb-3ba8150d339a.png"><img src="https://user-images.githubusercontent.com/170405/243440219-65eaf5fa-6a37-49e4-b82d-a0b57ad69ba9.png"></a>|<a target="_blank" href="https://user-images.githubusercontent.com/170405/243439353-c07ed02f-044e-4275-a255-50fe2ee29968.png"><img src="https://user-images.githubusercontent.com/170405/243440229-979f0d35-1001-46b3-9715-a4253e7d1a31.png"></a>|

## Changelog

The changelog could be found [here](https://github.com/GChristensen/deepfloyd_if_lab/wiki/Changelog).

## Frequently Asked Questions

>Q: It prints error messages several screens long. Is it normal?

A: Absolutely. DeepFloyd IF is an experimental library without a detailed user manual,
and you are running it in Jupyter notebooks.

>Q: It does not work, freezes, or crashes not displaying an error message. Are there any chances to make it work?

A: This may be anything, ranging from bugs to hardware incompatibility.
Unfortunately, you are out of luck, because it is impossible to determine what it is exactly.

>Q: It suddenly stopped working. What should I do?

A: Please try to restart the Jupyter Python kernel or the application.

>Q: It does not work even after I have restarted the application. What should I do next?

A: Please delete the `home/settings.json` file __*and*__ the entire `venv` folder.

>Q: I have enough VRAM, but encounter memory errors. Do I need a system upgrade?

A: If your computer does not meet the recommended requirements, it works near the limits of available 
resources. To run DeepFloyd IF you need as much free VRAM and system RAM as possible. Only the
recommended requirements allow to achieve more or less seamless experience.

>Q: Can I run this UI on a 8GB GPU?

A: The UI may run on a 8GB GPUs with 12GB of system RAM and the swap of the same size, but it may require constant restarts
due to the insufficient memory.

>Q: My generations look like halftone prints that were shredded and glued back by the pieces. How can I improve them?

A: Please check the guidance level. It might be too high. As a last resort, there is an option to not pass the
prompt to stage III. It is also possible to upscale the results of stage II using different upscaler.

>Q: Despite all my efforts, when doing inpainting I can't reproduce the effect of disappearing hat demonstrated
> at DeepFloyd IF GitHub page. I always get a static image and it looks blurry. Is there a way to improve this?

A: The [official demonstration](https://github.com/deep-floyd/IF#iv-zero-shot-inpainting) of DeepFloyd IF inpainting 
is quite misleading. Inpainting always produces a static image, and it looks blurry because this is how DeepFloyd IF
pipeline works. It reduces the source image to 64x64 pixels, inpaints there, and upscales it back. 
Probably there are bugs, or currently we do not know something that will allow us to obtain the same quality,
as it was demonstrated. 

>Q: How do I create an inpainting mask?

A: Currently, DeepFloyd IF Lab has no ability to interactively create a mask just by painting on the source image.
It is necessary to upload a black-and-white mask image along with the source image. It is possible
to create a mask image by painting over the source image on separate layers in your favorite graphical editor, 
or by directly transforming the current selection/alpha channel into the corresponding black-and-white image. 
Some editors have a macro system that allows to perform such operations in a single keystroke. Please refer to your 
editor user manual.

>Q: What are the advanced options for?

A: The advanced options allow to pass any supported argument values to the corresponding stages in the pipeline. 
For example, if you need to set aug_level to 0.2, specify aug_level=0.2 in one of these fields. The arguments are separated by commas.

>Q: I want to generate pretty anime girls with big cat ears using DeepFloyd IF, applying different character LoRAs.
How long do I need to wait until this functionality becomes available?

A: At first, it is necessary to wait until [A100s](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) will be 
available for $100/unit in used hardware stores, so anyone can get about a dozen of them to create LoRAs for DeepFloyd IF. 

Then someone should prepare a dataset of several millions of high-quality reasonably-labeled anime images without
problems with copyright and the availability of such content. The rest is to train an anime base model that is not 
overfitted and does things just right. Assuming that this model should be available free of charge, it is pretty
easy to estimate the time needed to wait for its general availability.

>Q: I want a feature X, REST API, and a plug-in system. Will they be implemented?

A: Probably, I would be able to work on this if I get enough donations to buy a new laptop with 128 GB of RAM and RTX 4090.
This may not happen in the next 10000 years.

>Q: For educational purposes only, I need to generate images of routine alien reproductive activity with AZC/BZC-chromosome
> alien individuals placed by the sides and CZC-chromosome individuals in between. Is it possible, given 
> the superior linguistic abilities of DeepFloyd IF?

A: Unfortunately, it is not possible. DeepFloyd IF was trained with some lacunas in the knowledge of such topics. Moreover, it has
a built-in safety filter that sometimes blurs random images which it considers too hot or hateful.
It seems, that DeepFloyd license prohibits any tinkering with this filter.

>Q: How this application is licensed?

A: This repository does not contain executable code derived from DeepFloyd IF and uses it as a library. 
It is licensed under BSD. Please remember that you may use DeepFloyd IF 1.0 only for personal research 
purposes due to its own license.

>Q: Wow! Your WebUI boosts my productivity fivefold. I know how it is hard to build
> software, and how much time is required to maintain it. Where I can send you some GWEI for thanks?

A: It is <a href="https://link.depay.com/AXgtLB6v1Iqx1Ufmnh7Hf">here</a>. Thank you.
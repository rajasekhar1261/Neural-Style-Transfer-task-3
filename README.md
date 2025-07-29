# Neural Style Transfer (optimization method) 💻 + 🎨 = ❤️

This repo contains a concise PyTorch implementation of the original NST paper (:link: Gatys et al.).

It's an accompanying repository for this video series on YouTube.

<img width="480" height="360" alt="image" src="https://github.com/user-attachments/assets/4b41efe2-1227-4ca6-9dc7-687e04931ec1" />

# What is NST algorithm?

The algorithm transfers style from one input image (the style image) onto another input image (the content image) using CNN nets (usually VGG-16/19) and gives a composite, stylized image out which keeps the content from the content image but takes the style from the style image.

<img width="750" height="500" alt="image" src="https://github.com/user-attachments/assets/9244cb6f-f11a-4fd8-9f98-42f64156a9f4" />
<img width="750" height="500" alt="image" src="https://github.com/user-attachments/assets/7b33caee-eb17-4a00-b8a8-d085275b2d8a" /> 

# Why yet another NST repo?

It's the cleanest and most concise NST repo that I know of + it's written in PyTorch! ❤️

Most of NST repos were written in TensorFlow (before it even had L-BFGS optimizer) and torch (obsolete framework, used Lua) and are overly complicated often times including multiple functionalities (video, static image, color transfer, etc.) in 1 repo and exposing 100 parameters over command-line (out of which maybe 5 or 6 may actually be used on a regular basis).

# Examples

Transfering style gives beautiful artistic results:

<img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/3fcf5ebd-7fd9-45e7-bfc2-dbeb93547edc" /> <img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/23eef2a6-f67d-409e-aafd-2b9c6a42f157" /> <img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/43aaf45c-1b73-4fe5-84b5-4c04cc4f7f73" />
<img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/4bbe066b-fd90-42bc-aad9-0ff657a0ca80" /> <img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/9610eb64-e6de-4c2b-9716-af3a0f7b632a" /> <img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/4226ac16-bb0e-409d-8a61-bcc724fa2826" />     

And here are some results coupled with their style:

<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/f2609203-ceac-415c-bc2c-7da8f098a10e" /> <img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/8a69fbfb-dc7d-449d-8c9f-21c0d795da1c" />
<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/67e65a55-049e-4a8d-8563-6d32d60f3611" /> <img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/19cb5abb-66c3-4dad-ab7d-70014740f106" />
<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/cdf0645b-1114-4403-b747-e5260243721a" /> <img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/480994f5-fb18-4cce-be26-8d3e594a145e" />
<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/2a922fb0-ad62-4dd5-af81-909f7863fe04" /> <img width="400" height="333" alt="image" src="https://github.com/user-attachments/assets/631c67ca-158c-4275-894b-05a5babe4b8c" />

Note: all of the stylized images were produced by me (using this repo), credits for original image artists are given bellow.

# Content/Style tradeoff

Changing style weight gives you less or more style on the final image, assuming you keep the content weight constant.
I did increments of 10 here for style weight (1e1, 1e2, 1e3, 1e4), while keeping content weight at constant 1e5, and I used random image as initialization image.

<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/9daf69d8-9f0d-45ea-981f-7aef1322db58" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/24b67cf2-4bbf-4d47-8eab-3e19ef9e3aef" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/a2df3c4c-8a2e-46bf-98c5-409785d38238" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/c4fc4551-ecd9-483d-ba2c-adca1b46baf2" />

# Impact of total variation (tv) loss

Rarely explained, the total variation loss i.e. it's corresponding weight controls the smoothness of the image.
I also did increments of 10 here (1e1, 1e4, 1e5, 1e6) and I used content image as initialization image.

<img width="350" height="233" alt="image" src="https://github.com/user-attachments/assets/1cce8a20-8c8c-4d37-8b33-d83c02e70238" /> <img width="350" height="233" alt="image" src="https://github.com/user-attachments/assets/0dabd2d5-5d68-48d3-a46e-9d19560ff3e8" /> <img width="350" height="233" alt="image" src="https://github.com/user-attachments/assets/b225af84-0210-42ea-9625-6eb860808b1c" /> <img width="350" height="233" alt="image" src="https://github.com/user-attachments/assets/324f7596-ad5b-4861-ae33-3bfd9bd2da33" />

# Optimization initialization

Starting with different initialization images: noise (white or gaussian), content and style leads to different results.
Empirically content image gives the best results as explored in this research paper also.
Here you can see results for content, random and style initialization in that order (left to right):

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/99130af1-dce8-404c-82ff-46bcbb91d559" /> <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/7c3f3372-8da8-4568-9a8e-fa274fea777f" /> <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/e53781fb-1e88-466a-a0cf-8f4dbc6db301" />

You can also see that with style initialization we had some content from the artwork leaking directly into our output.

# Famous "Figure 3" reconstruction

Finally if I haven't included this portion you couldn't say that I've successfully reproduced the original paper (laughs in Python):

 <img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/2010298c-682b-49fa-a0e8-a1bede31abc0" /> <img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/42c2b132-9278-4344-8715-7744fc910c03" />
<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/c219859b-ad6b-4009-8aae-642eff3d8671" /> <img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/9d11b630-28b9-4085-ad12-cd6d115ee819" />
<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/36b6e9da-8eeb-4f92-a7b3-051cc3712210" /> <img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/977dcaa7-17a7-40bf-8a19-610806f8d845" />

I haven't give it much effort results can be much nicer.

# Content reconstruction

If we only use the content (perceptual) loss and try to minimize that objective function this is what we get (starting from noise):

<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/81dcc9b0-ad7e-46a9-b434-fa91a5402faa" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/b2b03364-c5e6-4f7a-951a-e81e8d950f8b" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/558742be-6a09-4a85-9651-39efacd5a278" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/9d9da58d-6e36-4680-bda9-2a4a83382b8e" />

In steps 0, 26, 70 and 509 of the L-BFGS numerical optimizer, using layer relu3_1 for content representation.
Check-out this section if you want to play with this.

# Style reconstruction

We can do the same thing for style (on the left is the original art image "Candy") starting from noise:

<img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/78bbcd54-24e5-47d5-a7bc-4e33ba172b03" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/3d0df25a-df69-4f3a-aba6-fb44458177c1" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/88e4877a-c7ad-405b-9b69-b19377e4ccc9" /> <img width="250" height="250" alt="image" src="https://github.com/user-attachments/assets/60f32641-0eca-4fa6-ab95-ec1032e5889b" />

In steps 45, 129 and 510 of the L-BFGS using layers relu1_1, relu2_1, relu3_1, relu4_1 and relu5_1 for style representation.

# Setup

1.Open Anaconda Prompt and navigate into project directory cd path_to_repo
2.Run conda env create (while in project directory)
3.Run activate pytorch-nst

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies.

PyTorch package will pull some version of CUDA with it, but it is highly recommended that you install system-wide CUDA beforehand, mostly because of GPU drivers. I also recommend using Miniconda installer as a way to get conda on your system.

Follow through points 1 and 2 of this setup and use the most up-to-date versions of Miniconda (Python 3.7) and CUDA/cuDNN. (I recommend CUDA 10.1 as it is compatible with PyTorch 1.4, which is used in this repo, and newest compatible cuDNN)

# Usage

1.Copy content images to the default content image directory: /data/content-images/
2.Copy style images to the default style image directory: /data/style-images/
3.Run python neural_style_transfer.py --content_img_name <content-img-name> --style_img_name <style-img-name>

It's that easy. For more advanced usage take a look at the code it's (hopefully) self-explanatory (if you speak Python ^^).

Or take a look at this accompanying YouTube video, it explains how to use this repo in greater detail.

Just run it! So that you can get something like this: ❤️

<img width="719" height="500" alt="image" src="https://github.com/user-attachments/assets/650cb881-252c-4570-9f66-baafc957cf60" />

# Debugging/Experimenting

Q: L-BFGS can't run on my computer it takes too much GPU VRAM?
A: Set Adam as your default and take a look at the code for initial style/content/tv weights you should use as a start point.

Q: Output image looks too much like style image?
A: Decrease style weight or take a look at the table of weights (in neural_style_transfer.py), which I've included, that works.

Q: There is too much noise (image is not smooth)?
A: Increase total variation (tv) weight (usually by multiples of 10, again the table is your friend here or just experiment yourself).

# Reconstruct image from representation

I've also included a file that will help you better understand how the algorithm works and what the neural net sees.
What it does is that it allows you to visualize content (feature maps) and style representations (Gram matrices).
It will also reconstruct either only style or content using those representations and corresponding model that produces them.

Just run this:
reconstruct_image_from_representation.py --should_reconstruct_content <Bool> --should_visualize_representation <Bool>

And that's it! --should_visualize_representation if set to True will visualize these for you
--should_reconstruct_content picks between style and content reconstruction

Here are some feature maps (relu1_1, VGG 19) as well as a Gram matrix (relu2_1, VGG 19) for Van Gogh's famous starry night:

<img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/700376ad-70da-43b6-8405-be4633770974" /> <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/a53cdd94-9fa0-4b97-bc83-4368ecb64f85" /> <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/83b14b47-a855-4779-8ef5-fcc603e5a988" /> <img width="128" height="128" alt="image" src="https://github.com/user-attachments/assets/94f32263-4ff9-4257-aed7-8dfcd0fecfd8" />

No more dark magic.

# Acknowledgements

I found these repos useful: (while developing this one)

fast_neural_style (PyTorch, feed-forward method)
neural-style-tf (TensorFlow, optimization method)
neural-style (TensorFlow, optimization method)
I found some of the content/style images I was using here:

style/artistic images
awesome figures pic
awesome bridge pic
Other images are now already classics in the NST world.

# Citation

If you find this code useful for your research, please cite the following:

@misc{Gordić2020nst,
  author = {Gordić, Aleksa},
  title = {pytorch-neural-style-transfer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-neural-style-transfer}},
}

# Connect with me

If you'd love to have some more AI-related content in your life 🤓, consider:

Subscribing to my YouTube channel The AI Epiphany 🔔
Follow me on LinkedIn and Twitter 💡
Follow me on Medium 📚 ❤️

# Licence

License: MIT

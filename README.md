## Neural Style Transfer (optimization method) :computer: + :art: = :heart:
This repo contains a concise PyTorch implementation of the original NST paper (:link: [Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)).

It's an accompanying repository for [this video series on YouTube](https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608).

<p align="left">
<a href="https://www.youtube.com/watch?v=S78LQebx6jo" target="_blank"><img src="https://img.youtube.com/vi/S78LQebx6jo/0.jpg" 
alt="NST Intro" width="480" height="360" border="10" /></a>
</p>

### What is NST algorithm?
The algorithm transfers style from one input image (the style image) onto another input image (the content image) using CNN nets (usually VGG-16/19) and gives a composite, stylized image out which keeps the content from the content image but takes the style from the style image.

<p align="center">
<img width="750" height="500" alt="image" src="https://github.com/user-attachments/assets/23125073-7480-4096-8ce6-139a16df023c" />
<img width="344" height="500" alt="image" src="https://github.com/user-attachments/assets/e907eaa1-7696-42f3-8d00-114808e57a4d" />
</p>

### Why yet another NST repo?
It's the **cleanest and most concise** NST repo that I know of + it's written in **PyTorch!** :heart:

Most of NST repos were written in TensorFlow (before it even had L-BFGS optimizer) and torch (obsolete framework, used Lua) and are overly complicated often times including multiple functionalities (video, static image, color transfer, etc.) in 1 repo and exposing 100 parameters over command-line (out of which maybe 5 or 6 may actually be used on a regular basis).

## Examples

Transfering style gives beautiful artistic results:

<p align="center">
<img width="270" height="333" alt="image" src="https://github.com/user-attachments/assets/14985e3e-733a-45ad-8c42-4677b8f5ab76" />
<img width="270" height="333" alt="image" src="https://github.com/user-attachments/assets/97d86618-d2fc-4678-8944-75ceb770121e" />
<img width="270" height="333" alt="image" src="https://github.com/user-attachments/assets/a012f5df-b72d-43aa-ae0c-f7df7198f837" />

<img width="270" height="400" alt="image" src="https://github.com/user-attachments/assets/7bac308d-be77-4652-a741-49b2bac967d4" />
<img width="270" height="400" alt="image" src="https://github.com/user-attachments/assets/1f371dd2-60db-4745-8ee7-71a10750752f" />
<img width="270" height="400" alt="image" src="https://github.com/user-attachments/assets/445ad6a1-bdcb-4206-b923-68ed1525f583" />
</p>

And here are some results coupled with their style:

<p align="center">
<img width="350" height="300" alt="image" src="https://github.com/user-attachments/assets/7229f57d-bfe8-4349-9bd9-ff58a489a421" />
<img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/6f57dd18-3262-4a32-9bd4-ecd59713bb3f" />

<img width="350" height="300" alt="image" src="https://github.com/user-attachments/assets/e853a6e5-5012-4c3e-a6b1-f32d5237aa19" />
<img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/cff8ab44-3669-47f6-a1b3-9eecb33ffb11" />

<img width="350" height="300" alt="image" src="https://github.com/user-attachments/assets/b75fa45d-298f-4e7e-a4a7-af10140ebc5f" />
<img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/86004ebf-3374-4059-83b9-7d4453d70d9e" />

<img width="350" height="300" alt="image" src="https://github.com/user-attachments/assets/8bfaba48-e26e-4e09-8155-533ffac9cdae" />
<img width="350" height="250" alt="image" src="https://github.com/user-attachments/assets/26977fae-8961-466b-83a4-199571d24e83" />
</p>

*Note: all of the stylized images were produced by me (using this repo), credits for original image artists [are given bellow](#acknowledgements).*

### Content/Style tradeoff

Changing style weight gives you less or more style on the final image, assuming you keep the content weight constant. <br/>
I did increments of 10 here for style weight (1e1, 1e2, 1e3, 1e4), while keeping content weight at constant 1e5, and I used random image as initialization image. 

<p align="center">
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/26e08abe-c682-462b-b9f0-e6a20761d7a9" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/f4c332bb-e2fc-4715-bc85-358547f4a10d" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/6e8c56c5-0355-4ecd-a5d5-139a0ac845df" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/18ebdd7c-4097-40d4-8e0c-f8ee5e029bff" />
</p>

### Impact of total variation (tv) loss

Rarely explained, the total variation loss i.e. it's corresponding weight controls the smoothness of the image. <br/>
I also did increments of 10 here (1e1, 1e4, 1e5, 1e6) and I used content image as initialization image.

<p align="center">
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/258c4634-30fe-4dc9-a713-b9776239d85e" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/19cf1b07-3716-4e2b-9663-190ef7f30b61" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/1412a85f-ad59-4553-8edc-1076eb556439" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/0ae0a65c-de9f-44eb-ba79-e9e958daa19b" />

</p>

### Optimization initialization

Starting with different initialization images: noise (white or gaussian), content and style leads to different results. <br/>
Empirically content image gives the best results as explored in [this research paper](https://arxiv.org/pdf/1602.07188.pdf) also. <br/>
Here you can see results for content, random and style initialization in that order (left to right):

<p align="center">
<img width="300" height="350" alt="image" src="https://github.com/user-attachments/assets/f18a44c4-0cd1-430d-b73f-8d990a692e9e" />
<img width="300" height="350" alt="image" src="https://github.com/user-attachments/assets/d2d66812-071f-4385-accb-cbd42686717d" />
<img width="300" height="350" alt="image" src="https://github.com/user-attachments/assets/451c68a9-8d9c-4023-a579-dbc70bb23742" />
</p>

You can also see that with style initialization we had some content from the artwork leaking directly into our output.

### Famous "Figure 3" reconstruction

Finally if I haven't included this portion you couldn't say that I've successfully reproduced the [original paper]((https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)) (laughs in Python):

<p align="center">
<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/649c158a-4b14-426e-92ca-04d55cc91379" />
<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/db31bc78-dd32-49c4-b29f-2c414de04e75" />
<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/2d91ff01-7a62-4a35-9ea5-eebdfd93ef29" />

<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/1a6c8505-6699-46d2-b707-4d2d43a37741" />
<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/5f471630-a7e6-41bb-8a27-2b60852c435a" />
<img width="320" height="240" alt="image" src="https://github.com/user-attachments/assets/05bdb376-9b10-4ce2-93d1-f4b86658c0d7" />
</p>

I haven't give it much effort results can be much nicer.

### Content reconstruction

If we only use the content (perceptual) loss and try to minimize that objective function this is what we get (starting from noise):

<p align="center">
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/f2e06e30-a8ea-4281-9292-c53b6d468da6" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/0afc75c0-f841-419a-9e1b-76ccca47f2e4" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/534eb209-5144-4d64-b2d4-aaf30c7801c6" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/a6039933-37a6-4e2f-a09d-aa196432cf70" />
</p>

In steps 0, 26, 70 and 509 of the L-BFGS numerical optimizer, using layer relu3_1 for content representation.<br/> 
Check-out [this section](#reconstruct-image-from-representation) if you want to play with this.

### Style reconstruction

We can do the same thing for style (on the left is the original art image "Candy") starting from noise:

<p align="center">
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/605fbf26-4725-4b69-bc2f-25db1b9a13be" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/d8fe4cda-54a8-4f5f-bbf9-b31742c9ab86" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/d268a62f-f64a-41df-800a-1a82e1b341c3" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/51387361-0a41-4035-991e-bcc4f07c6680" />
</p>

In steps 45, 129 and 510 of the L-BFGS using layers relu1_1, relu2_1, relu3_1, relu4_1 and relu5_1 for style representation.

## Setup

1. Open Anaconda Prompt and navigate into project directory `cd path_to_repo`
2. Run `conda env create` (while in project directory)
3. Run `activate pytorch-nst`

That's it! It should work out-of-the-box executing environment.yml file which deals with dependencies.

-----

PyTorch package will pull some version of CUDA with it, but it is highly recommended that you install system-wide CUDA beforehand, mostly because of GPU drivers. I also recommend using Miniconda installer as a way to get conda on your system. 

Follow through points 1 and 2 of [this setup](https://github.com/Petlja/PSIML/blob/master/docs/MachineSetup.md) and use the most up-to-date versions of Miniconda (Python 3.7) and CUDA/cuDNN.
(I recommend CUDA 10.1 as it is compatible with PyTorch 1.4, which is used in this repo, and newest compatible cuDNN)

## Usage

1. Copy content images to the default content image directory: `/data/content-images/`
2. Copy style images to the default style image directory: `/data/style-images/`
3. Run `python neural_style_transfer.py --content_img_name <content-img-name> --style_img_name <style-img-name>`

It's that easy. For more advanced usage take a look at the code it's (hopefully) self-explanatory (if you speak Python ^^).

Or take a look at [this accompanying YouTube video](https://www.youtube.com/watch?v=XWMwdkaLFsI), it explains how to use this repo in greater detail.

Just run it! So that you can get something like this: :heart:
<p align="center">
<img width="719" height="500" alt="image" src="https://github.com/user-attachments/assets/2ab9f3f2-9611-4465-8fff-1abd1390ff2b" />
</p>

### Debugging/Experimenting

Q: L-BFGS can't run on my computer it takes too much GPU VRAM?<br/>
A: Set Adam as your default and take a look at the code for initial style/content/tv weights you should use as a start point.

Q: Output image looks too much like style image?<br/>
A: Decrease style weight or take a look at the table of weights (in neural_style_transfer.py), which I've included, that works.

Q: There is too much noise (image is not smooth)?<br/>
A: Increase total variation (tv) weight (usually by multiples of 10, again the table is your friend here or just experiment yourself).

### Reconstruct image from representation

I've also included a file that will help you better understand how the algorithm works and what the neural net sees.<br/>
What it does is that it allows you to visualize content **(feature maps)** and style representations **(Gram matrices)**.<br/>
It will also reconstruct either only style or content using those representations and corresponding model that produces them. <br/> 

Just run this:<br/>
`reconstruct_image_from_representation.py --should_reconstruct_content <Bool> --should_visualize_representation <Bool>`
<br/><br/>
And that's it! --should_visualize_representation if set to True will visualize these for you<br/>
--should_reconstruct_content picks between style and content reconstruction

Here are some feature maps (relu1_1, VGG 19) as well as a Gram matrix (relu2_1, VGG 19) for Van Gogh's famous [starry night](https://en.wikipedia.org/wiki/The_Starry_Night):

<p align="center">
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/121e9e20-11c5-41ca-9d5a-1b3cc089a9fc" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/f92e8118-edc8-4a65-9fde-866a2ebdf294" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/53683f91-590f-4d17-8186-f536315082b6" />
<img width="150" height="250" alt="image" src="https://github.com/user-attachments/assets/420905e5-77f6-4c89-8ef8-e3eeaad3c9ca" />
</p>

No more dark magic.

## Acknowledgements

I found these repos useful: (while developing this one)
* [fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style) (PyTorch, feed-forward method)
* [neural-style-tf](https://github.com/cysmith/neural-style-tf/) (TensorFlow, optimization method)
* [neural-style](https://github.com/anishathalye/neural-style/) (TensorFlow, optimization method)

I found some of the content/style images I was using here:
* [style/artistic images](https://www.rawpixel.com/board/537381/vincent-van-gogh-free-original-public-domain-paintings?sort=curated&mode=shop&page=1)
* [awesome figures pic](https://www.pexels.com/photo/action-android-device-electronics-595804/)
* [awesome bridge pic](https://www.pexels.com/photo/gray-bridge-and-trees-814499/)

Other images are now already classics in the NST world.

## Citation

If you find this code useful for your research, please cite the following:

```
@misc{Gordić2020nst,
  author = {Gordić, Aleksa},
  title = {pytorch-neural-style-transfer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-neural-style-transfer}},
}
```

## Connect with me

If you'd love to have some more AI-related content in your life :nerd_face:, consider:
* Subscribing to my YouTube channel [The AI Epiphany](https://www.youtube.com/c/TheAiEpiphany) :bell:
* Follow me on [LinkedIn](https://www.linkedin.com/in/aleksagordic/) and [Twitter](https://twitter.com/gordic_aleksa) :bulb:
* Follow me on [Medium](https://gordicaleksa.medium.com/) :books: :heart:

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/pytorch-neural-style-transfer/blob/master/LICENCE)

#!/usr/bin/env python
# coding: utf-8

# # text2voxels v0.2.1
# 
# by [@nev#4905](https://twitter.neverix.io/)

# In[ ]:


#@title mount Google Drive
# from google.colab import drive
# do_it = True  #@param {type: "boolean"}
# if do_it:
#     drive.mount('/content/drive')


# In[ ]:


#@title create CLIP
# get_ipython().system('pip install torch_optimizer > /dev/null 2> /dev/null')
# get_ipython().system('pip install clip-anytorch > /dev/null 2> /dev/null')
import clip


clip_version = "ViT-B/16"  #@param ["ViT-B/16", "ViT-B/32", "RN50", "RN50x4"] {type: "string", allow-input: true}
model, preprocess = clip.load(clip_version, jit=False)


# In[ ]:


#@title noise
#@markdown like in the dream fields paper

#@markdown stolen from https://gist.github.com/ac1b097753f217c5c11bc2ff396e0a57

# ported from https://github.com/pvigier/perlin-numpy/blob/master/perlin2d.py
import random
import torch
import math


#@markdown lowering to 0.2 sometimes improves the results
noise_level = 0.5  #@param {type: "number"}


def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
    
    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
    
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    noise *= random.random() - noise_level  # haha
    noise += random.random() - noise_level  # haha x2
    return noise


# In[ ]:


# @title does something
import gc
import io
import os
import time
import torch
import random
import requests
import numpy as np
import torchvision
from PIL import Image
from math import ceil
import torch_optimizer
from base64 import b64encode
from ipywidgets import Output
from IPython.display import HTML
from more_itertools import chunked
from tqdm.auto import trange, tqdm
from subprocess import Popen, PIPE
from matplotlib import pyplot as plt
plt.show_ = lambda **kwargs: plt.pause(0.0000001)  # plt.show
clear_output = lambda **kwargs: plt.clf()
# from IPython.display import display, clear_output


gc.collect()
torch.cuda.empty_cache()
model.requires_grad_(False)
#@markdown set to zero for no seed
seed =   24#@param {type: "integer"}
#@markdown ## video settings
#@markdown image size, must be divisible by 28
w =   224#@param {type: "integer"}
#@markdown image size for CLIP, can't be bigger than w. 252 also only works with ViT-B/32
cutout_w =   "224"#@param [224, 252] {type: "string"}
#@markdown frames per second
fps = 15  #@param {type: "integer"}
#@markdown if empty, make the name of the output video the same as the text
out_path = ""  #@param {type: "string"}
#@markdown download the resulting video automatically
download = False  #@param {type: "boolean"}
#@markdown ## text
#@markdown prompts with capital letters work better
text = "a black swan"  #@param ["dream landscape by Van Gogh", "a black swan", "a flying whale", "a white rabbit", "a beautiful bonsai tree", "beautiful pine trees", "a cute creature", "chtulhu is watching", "a frog", "a robot dog. a robot in the shape of a dog", "a carrot", "personal computer", "golden pyramid", "golden snitch", "a new planet ruled by ophanims", "a blue cat", "a burning potato", "an eggplant", "a purple mouse", "a ninja", "a green teapot", "a black rabbit", "an avocado armchair", "smoking a joint", "3D old-school telephone", "a wooden chair", "spider-man figure", "a 3D monkey #pixelart", "a minecraft grass block", "minecraft creeper", "minecraft landscape", "a shining star"] {allow-input: true}
#@markdown capitalize the first character automatically?
capitalize = True  #@param {type: "boolean"}
#@markdown capitalize every word automatically?
capitalize_words = False  #@param {type: "boolean"}
#@markdown capitalize every letter after a number (3d -> 3D)
capitalize_number = True  #@param {type: "boolean"}
#@markdown rename the output video if it exists
rename_out = True  #@param {type: "boolean"}
#@markdown ## learning settings
#@markdown load a checkpoint saved before, leave empty for no checkpoint
load_checkpoint = ""  #@param {type: "string"}
#@markdown load checkpoints from google drive
load_gdrive = False  #@param {type: "boolean"}
#@markdown ignore file not found error and start from scratch if the checkpoint file was not found
ignore_err = True  #@param {type: "boolean"}
#@markdown number of steps to train for
train_steps =    720#@param {type: "integer"}
#@markdown stop training after `time_stop` seconds. negative values are ignored

#@markdown stops after two minutes by default.

#@markdown you can stop the generation process manually
time_stop =    -1#@param {type: "number"}
#@markdown optimizationparameters

#@markdown increasing the batch size or gradient accumulation makes the results a little smoother and gets rid of duplicates

#@markdown training runs with gradient accumulation (1-2) look better
grad_acc = 2#@param {type: "integer"}
train_batch =   2#@param {type: "integer"}
#@markdown learning rate, decreasing it makes the optimization slower but slightly more detailed, anything below 1e-2 is too slow
lr =   1e-1#@param {type: "number"}
#@markdown FP16, reduces memory usage but not fully tested
fp16 = False  #@param {type: "boolean"}
#@markdown optimizer name, you can use any from the library [torch-optimizer](https://github.com/jettify/pytorch-optimizer)

#@markdown these might use more memory, so decrease rendering steps and batch size if an error appears

#@markdown the optimizers are ordered by performance decreasing

#@markdown when using Adam, increase the gradient accumulation to ~8

#@markdown MADGRAD is pretty good but uses more memory
optimizer_name = "MADGRAD"  #@param ["Adam", "MADGRAD"] {type: "string", allow-input: true}
#@markdown ## rendering settings
#@markdown size of the box
extent =   1#@param {type: "number"}  # 1.28 # 1.3  # 1  # 1.5
#@markdown bilinear interpolation (uses 8x more memory, makes slightly smoother images)
use_weights = False  #@param {type: "boolean"}
#@markdown near and far planes for the camera
near =  1#@param {type: "number"}
far =   5#@param {type: "number"}
#@markdown size of the FOV plane. 1 for 90 degrees, more for more
fov_plane =  0.75 #@param {type: "number"}
#@markdown side of the box. 128 is 4x faster, but it might be lower quality
block_size = 256
#@markdown density at the edges
mask_value = 0  #@param {type: "number"}
#@markdown default camera offset and angle
offset =   3#@param {type: "number"}
angle = 0  #@param {type: "number"}
#@markdown starting scale for the pyramid
scale_from =   1#@param {type: "integer"}
#@markdown how much the scale grows (exponentially)
scale_decay = 0.25  #@param {type: "number"}  # 0.5
#@markdown raise the scale while training? (not implemented)
scale_schedule = False  #@param  {type: "boolean"}
#@markdown initialization density
init_density = 0.05  #@param {type: "number"}
#@markdown initialization type (what the cube looks like in the beginning)
init_type = "uniform"  #@param ["uniform", "random", "spherical"] {type: "string"}
#@markdown (for spherical init) the power that the init is raised to
init_pow = 1  #@param {type: "number"}
#@markdown number of raycasting steps. raising this improves the resolution but makes the renderer use more memory
rendering_steps = 100  #@param {type: "integer"}
#@markdown background color
bg_color = 0.95  #@param {type: "number"}
#@markdown grayscale rendering
grayscale = False  #@param {type: "boolean"}
#@markdown cutoff density for export and sparse rendering
quantize_thresh = 0.2 #@param {type: "number"}
#@markdown palette image URL or local path

#@markdown colors will be extracted from this and used for the voxels

#@markdown warning: palettes slow down generation by around 2 times.
palette_path = ""  #@param ["", "https://cdn.pixabay.com/photo/2019/07/03/01/55/black-swan-4313444_1280.jpg", "https://pixahive.com/wp-content/uploads/2020/12/A-car-in-a-desert-236323-pixahive.jpg"] {type: "string", allow-input: true}
#@markdown number of colors for the palette
palette_colors =  6#@param {type: "integer"}
#@markdown ## objective settings
#@markdown the image prompt URL or local path. make blank if you don't want image prompting
img_path = ""  #@param {type: "string"}
#@markdown similarity to the image prompt, ignored if you use a blank image URL
mse_coeff = 0  #@param {type: "number"}
#@markdown image prompt plane, see fov_plane
mse_fov = 0.5  #@param {type: "number"}
#@markdown render a separate view for the image prompt
mse_single = False #@param {type: "boolean"}
#@markdown background of the view
mse_bg = 1.0 #@param {type: "number"}
#@markdown L2 regularization. reg_color: regularize RGB apart from density?
reg_coeff = 3  #@param {type: "number"}  # 1
reg_color = True  #@param {type: "boolean"}
#@markdown TV regularization, increase this to 4 or 5 to make the image smoother if it is too noisy
tv_coeff =   3#@param {type: "number"}  # 1050
#@markdown CLIP weight
clip_coeff = 20  #@param {type: "number"}  # 4
#@markdown CLIP loss type, might improve the results
loss_type = "cosine"  #@param ["spherical", "cosine"] {type: "string"}
#@markdown spherical regularization coefficient. making this higher "shrinks" the shape. this is preferred for making the image more coherent over tau_coeff
spherical_coeff =    35#@param {type: "number"}  # 100
#@markdown weighting for size of the virtual sphere. raise this to make the shape a little bigger
sphere_size =   20#@param {type: "number"}
#@markdown tau regularization from dream fields. tau_target limits the shape's visual size and the coefficient while tau_coeff makes the shape disappear faster
tau_coeff =   5#@param {type: "number"}
tau_target = 0.5  #@param {type: "number"}  # 0.18  # 0.25  # 0.2 * 0.5
#@markdown ranges for random azimuth, altitude and offset shifts (augmentations)
shuffle_ang =   10#@param {type: "number"}  # 1.28  # 10  # 0.1
shuffle_altitude = 0.0  #@param {type: "number"}  # 0.1
shuffle_offset =   0.1#@param {type: "number"}
shuffle_xy =   0.1#@param {type: "number"}  # 1.28  # 10  # 0.1
first_for = 3  #@param {type: "number"}
#@markdown ## timing settings
#@markdown how long to spin before training, showing the empty cube
spin_before =   30#@param {type: "integer"}
#@markdown how long to spin after training
spin_length = 60  #@param {type: "integer"}
#@markdown how many spins to perform after training
spin_number = 2  #@param {type: "integer"}
#@markdown number of frames to show the still picture for
still_frames = 30  #@param {type: "integer"}
#@markdown how many spins to perform while training
train_spins = 1  #@param {type: "integer"}
#@markdown rotate the visualization while training?
do_rotate = True  #@param {type: "boolean"}
#@markdown skip training when visualizing?
only_spin = False  #@param {type: "boolean"}
#@markdown show how the scene actually looks like or debug images?
spin_quality = True  #@param {type: "boolean"}
#@markdown display the image prompt?
display_img = False  #@param {type: "boolean"}
#@markdown raise an error after modifying settings (useful for debugging)
stop_after_settings = False  #@param {type: "boolean"}
#@markdown keep adding to the same video on every run. this is recommended for continuing an interrupted run
same_video = False  #@param {type: "boolean"}
#@markdown continue the optimization you stopped in the middle. this is recommended for continuing an interrupted run
same_color = False  #@param {type: "boolean"}
#@markdown note: you don't need these options if you're already loading from a checkpoint


try:
    frames
except NameError:
    frames = []

device = "cuda" if torch.cuda.is_available() else "cpu"

if not seed:
    seed = random.getrandbits(8)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# fetch function from a diffusion notebook
# used for geting image prompts and palettes
def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


# shape rotator. moves and offsets rays
def prepare(xyd, offset_z, angle_x, angle_y=0, offset_x=0, offset_y=0, batch=1):
    tensorize = lambda x: x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) if isinstance(x, torch.Tensor) else x
    xyd = xyd.repeat(batch, 1, 1, 1, 1)
        
    offset_x = tensorize(offset_x)
    offset_y = tensorize(offset_y)
    offset_z = tensorize(offset_z)
    
    xyd[..., 0] += offset_x
    xyd[..., 1] += offset_y
    xyd[..., 2] -= offset_z
    
    a = torch.atan2(xyd[..., 2], xyd[..., 1])
    f = (xyd[..., [1, 2]] ** 2).sum(dim=-1) ** 0.5
    angle_y = tensorize(angle_y)
    a += angle_y
    xyd = torch.stack((xyd[..., 0], torch.cos(a) * f, torch.sin(a) * f), dim=-1)
    
    a = torch.atan2(xyd[..., -1], xyd[..., 0])
    f = (xyd[..., [0, -1]] ** 2).sum(dim=-1) ** 0.5
    angle_x = tensorize(angle_x)
    a += angle_x
    xyd = torch.stack((torch.cos(a) * f, xyd[..., 1], torch.sin(a) * f), dim=-1)
    
    return xyd


# clamp with gradients
def cl(x):
    return torch.relu(1 - torch.relu(1 - x))


# simple 3D voxel renderer. very inefficient, no filtering
@torch.jit.script
def render(color, xyd,
           extent: float = extent,
           bg_color: float = bg_color,
           use_weights: bool = use_weights,
           mask_value: float = mask_value,
           return_depth: bool = False):
    color = torch.nn.functional.pad(color[1:-1, 1:-1, 1:-1],
                                    (0, 0, 1, 1, 1, 1, 1, 1),
                                    value=mask_value)
    color = cl(color)
    # idk how to do bilinear interpolation in 3D so this entire section is just that
    with torch.no_grad():
        xyd = xyd / 2 / extent + 0.5
        xyd = xyd.clamp(0, 1)
        xyd *= torch.tensor(color.shape[:-1]).to(color.device) - 1
        rounds = [xyd]
        weights = [torch.ones_like(xyd[..., -1])]
        for dim in range(xyd.shape[-1] * int(use_weights)):
            new_rounds = []
            new_weights = []
            for r, m in zip(rounds, weights):
                d = r[..., dim] - r[..., dim].floor()
                r1 = r.clone()
                r1[..., dim] = r[..., dim].floor()
                new_rounds.append(r1)
                new_weights.append((1 - d) * m)
                r2 = r.clone()
                r2[..., dim] = r[..., dim].ceil()
                new_rounds.append(r2)
                new_weights.append(d * m)
            rounds = new_rounds
            weights = new_weights
        rounds = torch.stack(rounds, dim=-2).long()
        weights = torch.stack(weights, dim=-1)
        for t in range(rounds.shape[-1]):  # [2, 1]
            rounds[..., :t] *= color.shape[t]
        rounds = rounds.sum(dim=-1)
    # this is the actual renderer
    color = color.view((-1, color.shape[-1]))[rounds.ravel(), :].view(rounds.shape + (color.shape[-1],))
    color = color * weights[..., None]
    color = color.sum(dim=-2)
    density = color[..., -1]
    lg = torch.cat((density[..., :1] * 0, torch.log(1 - density[..., 1:])), dim=-1)
    resid = torch.exp(torch.cumsum(lg, dim=-1))
    density = density * resid
    rgb = (color * density[..., None]).sum(dim=-2) + bg_color * (1 - density.sum(dim=-1))[..., None]
    if not return_depth:
        return rgb, density.sum(dim=-1)
    else:
        return rgb, density


# total variation regularizer
def tv(x):
    return ((x[1:] - x[:-1]) ** 2).mean() + ((x[:, 1:] - x[:, :-1]) ** 2).mean() + ((x[:, :, 1:] - x[:, :, :-1]) ** 2).mean()


# blur the image for free pyramids
def interpolate(color, scale_from=scale_from, scale_decay=scale_decay, grayscale=grayscale):
    new_color = color.permute(3, 0, 1, 2).unsqueeze(0)
    res = torch.zeros_like(new_color)
    s = block_size / (2 ** scale_from)
    p = 1 / (2 ** scale_from)
    j = 0
    total = 0
    while s > 0:
        scale = 2 ** (scale_decay * j)
        res = res + scale * torch.nn.functional.interpolate(
            torch.nn.functional.interpolate(
                new_color, scale_factor=p, mode="trilinear", align_corners=False),
                size=new_color.shape[-3:], mode="trilinear", align_corners=False)
        total += scale
        s, p, j = s // 2, p / 2, j + 1  # binary pyramid
    res = res / total
    res = res[0].permute(1, 2, 3, 0)
    if grayscale:
        res = torch.cat((torch.stack((res[..., :-1].mean(dim=-1),) * 3, dim=-1), res[..., -1:]), dim=-1)
    if palette is not None:
        colors = res[..., :3]
        with torch.inference_mode():
            palette_dist = ((colors[..., None, :] - palette) ** 2).sum(dim=-1)
            colors_chosen = palette_dist.argmin(dim=-1)
            del palette_dist
            new_colors = palette[colors_chosen]
            del colors_chosen
            shift = new_colors - colors
            del new_colors
        # z+q trick from pixray
        colors = colors + shift
        del shift
        res = torch.cat((colors, res[..., 3:]), dim=-1)
    return res


def setup():
    # bad but it works
    global color, src_array, out_path, xyd, src, mse_coeff, mse_single, palette
    
    if os.path.exists("source.png"):
        src = Image.open("source.png").resize((w, w)).convert("RGB")
        print("Image prompt:")
        print("Use your imagination :)")  # display(src)
        src_array = torch.from_numpy(np.asarray(src) / 255).to(device)
    else:
        src_array = torch.zeros((w, w, 3), device=device)
        mse_coeff = 0.0
        mse_single = False
    if os.path.exists("palette.png"):
        palette = (Image.open("palette.png")
                        .convert("RGB"))
        palette = palette.resize((int(w * pallete.size[0] / palette.size[1]),
                                  w) if palette.size[1] > palette.size[0] else (
                                      w, int(w * palette.size[1] / palette.size[0])))
        print("Palette:")
        print("Use your imagination :)")  # display(palette)
        palette = np.asarray(palette).reshape((-1, 3)) / 255
        from sklearn.cluster import KMeans
        palette = KMeans(n_clusters=palette_colors
                                ).fit(palette).cluster_centers_
        palette = torch.from_numpy(palette).to(device)
    else:
        palette = None
    with torch.no_grad():
        y, x = torch.meshgrid(((torch.arange(w, device=device) / w * 2 - 1) * fov_plane,) * 2)
        z = torch.linspace(near, far, rendering_steps, device=device)
        xy = torch.stack((x, y), dim=-1)
        d = z.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        xyd = torch.cat((xy
                            .unsqueeze(-2).unsqueeze(0)
                            .repeat(1, 1, 1, d.shape[-2], 1),
                            torch.ones_like(d).unsqueeze(0)
                            .repeat(1, w, w, 1, 1)
    ), dim=-1) * d

    frames = []
    voxels = torch.stack(torch.meshgrid(*((torch.arange(block_size, device=device) / block_size * 2 - 1,) * 3)), dim=-1)
    color = torch.cat((voxels, voxels[..., -1:]), dim=-1).clone()
    color[..., :3] = torch.rand_like(color[..., :3])
    if init_type == "uniform":
        color[..., -1] = torch.ones_like(color[..., -1]) ** init_pow
    elif init_type == "random":
        color[..., -1] = torch.rand_like(color[..., -1]) ** init_pow
    elif init_type == "spherical":
        color[..., -1] = 1 - ((color[..., :3] ** 2).mean(dim=-1) ** 0.5) ** init_pow
    else:
        raise NotImplementedError(f"Init type {init_type} not supported")
    color[..., -1] /= color[..., -1].max()
    color[..., -1] *= init_density
    color = torch.nn.Parameter(color.detach(), requires_grad=True)


# forward renderer
def spin(length=spin_length, total=None, start=0,
         progress_bar=True, clear=True, show=True, ignore_err=True,
         spins=1, interpolate=interpolate,
         return_depth=False, return_coords=False):
    if total is None:
        total = length
    # if show:
        # out = Output()
        # display(out)
    try:
        angles = list(range(start, start+length))
        tq = tqdm if progress_bar else lambda x: x
        for i in tq(list(chunked(angles, 4))):
            i = torch.tensor(i, dtype=torch.float32, device=device)
            i *= np.pi * 2 * spins / total
            with torch.inference_mode():
                res = interpolate(color)
                coords = prepare(xyd, offset, i, batch=len(i))
                pics, depths = render(
                    res,
                    coords,
                    return_depth=return_depth)
                pics = pics.cpu().numpy()
                depths = depths.cpu().numpy()
                coords = coords.cpu().numpy()
            for pic, depth, coord in zip(pics, depths, coords):
                img = Image.fromarray((pic[..., :3] * 255).astype(np.uint8))
                img = img,
                if return_depth:
                    img = img + (depth,)
                if return_coords:
                    img = img + (coord,)
                if len(img) == 1:
                    img = img[0]
                yield img
            
            if show:
                # with out:
                if clear:
                    clear_output(wait=True)
                plt.axis("off")
                plt.imshow(pics[0, ..., :3])
                plt.show_()
    except KeyboardInterrupt:
        if ignore_err:
            pass
        else:
            raise
    

def cutout(img, w=int(cutout_w)):
    y = random.randint(0, img.shape[0] - w)
    x = random.randint(0, img.shape[1] - w)
    return img[y:y+w, x:x+w]


# trainer
def train(text=text, frames=frames):
    global it, bar
    # out = Output()
    # display(out)
    with torch.no_grad():
        txt_emb = model.encode_text(clip.tokenize(text).to(device))
        txt_emb = torch.nn.functional.normalize(txt_emb, dim=-1)
    params = [color]
    try:
        optimizer_class = getattr(torch.optim, optimizer_name)
    except AttributeError:
         optimizer_class = getattr(torch_optimizer, optimizer_name)
    optimizer = optimizer_class(params, lr)
    # torch.optim.Adam(params, lr=lr)
    bar = trange(train_steps)
    loss_acc = 0
    acc_n = 0
    losses = []
    start_time = time.time()
    try:
        for it in bar:
            rot = (torch.randn(train_batch, device=device)) * shuffle_ang
            rot_y = (torch.randn(train_batch, device=device)) * shuffle_altitude
            offset_x = torch.randn(train_batch, device=device) * shuffle_xy
            offset_y = torch.randn(train_batch, device=device) * shuffle_xy
            res = interpolate(color)
            with torch.cuda.amp.autocast(enabled=fp16):
                img, d = render(res,  # color,
                    prepare(xyd, offset
                            + torch.rand(train_batch, device=device) * shuffle_offset,
                            rot, rot_y, offset_x=offset_x, offset_y=offset_y,
                            batch=train_batch), bg_color=0)
            img = img[..., :3]
            d = d.unsqueeze(-1)
            back = torch.zeros_like(img)
            s = back.shape
            for i in range(s[0]):
                for j in range(s[-1]):
                    n = random.choice([7, 14, 28])
                    back[i, ..., j] = rand_perlin_2d_octaves(s[1:-1], (n, n)).clip(-0.5, 0.5) + 0.5
            img = img + back * (1 - d)
            pics = img.detach().cpu().numpy()
            if not spin_quality:
                for pic in pics:
                    frames.append(Image.fromarray((pic * 255).astype(np.uint8)))
            
            
            if mse_single:
                xyd_ = prepare(xyd, offset, 0, batch=1)
                img_mse, _ = render(res,
                                    torch.cat((xyd_[..., :2] * mse_fov,
                                            xyd_[..., 2:]), dim=-1),
                                    bg_color=mse_bg)
            else:
                img_mse = img.clone()

            # with out:
            clear_output(wait=True) 
            plt.plot(losses)
            plt.show_()
            if not spin_quality:
                plt.axis("off")
                plt.imshow(pics[0, ..., :3])
                plt.show_()
            else:
                frames += list(spin(total=len(bar),
                                    start=it * int(do_rotate), length=1,
                                    progress_bar=False, clear=False,
                                    spins=train_spins))
            if mse_single and mse_coeff:
                plt.axis("off")
                plt.imshow(img_mse[0, ..., :3].detach().cpu().numpy())
                plt.show_()
            img_clip = torch.stack([cutout(x) for x in img], dim=0).permute(0, 3, 1, 2)
            img_clip = torchvision.transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img_clip)
            img_emb = model.encode_image(img_clip)
            img_emb = torch.nn.functional.normalize(img_emb, dim=-1)
            img_emb[torch.isnan(img_emb)] = 0
            txt_emb[torch.isnan(txt_emb)] = 0
            img_emb[torch.isinf(img_emb)] = 0
            txt_emb[torch.isinf(txt_emb)] = 0

            x, y, z = torch.meshgrid(*(((torch.arange(block_size) - block_size // 2) / (block_size // 2),) * 3))
            x, y, z = x * sphere_size, y * sphere_size, z * sphere_size
            sphere = (x ** 2 + y ** 2 + z ** 2).unsqueeze(0).repeat(train_batch, 1, 1, 1).to(device)
            sphere = sphere / sphere.max()
            
            spherical_loss = (sphere * (color[..., -1] ** 2) * torch.sign(color[..., -1])).mean()
            if loss_type == "spherical":
                clip_loss = (img_emb - txt_emb).norm(dim=-1).div(2).arcsin().pow(2).mul(2).mean()
            elif loss_type == "cosine":
                clip_loss = (1 - img_emb @ txt_emb.T).mean()
            else:
                raise NotImplementedError(f"CLIP loss type not supported: {loss_type}")
            mse_loss = ((img_mse[..., :3] - src_array.unsqueeze(0)) ** 2).mean()
            reg_loss = ((color if reg_color else color[..., -1:]) ** 2).mean()
            tv_loss = tv(res)
            tau_loss = d.mean().clamp(tau_target, 100)
            loss = (
                mse_loss * mse_coeff +
                reg_loss * reg_coeff + 
                tv_loss * tv_coeff + 
                clip_loss * clip_coeff +
                tau_loss * tau_coeff +
                spherical_loss * spherical_coeff)
            loss.backward()
            for param in params:
                param.grad.data[torch.isnan(param.grad.data)] = 0
                param.grad.data[torch.isinf(param.grad.data)] = 0
            loss_acc += loss.item()
            acc_n += 1
            acc_n += 1
            bar.set_description(f"loss: {loss_acc / max(acc_n, 1)}"
                                f" mse: {mse_loss.item()} reg: {reg_loss.item()}"
                                f" tv: {tv_loss.item()} clip: {clip_loss.item()}"
                                f" tau: {tau_loss.item()} spherical: {spherical_loss.item()}")
            if it % grad_acc == grad_acc - 1:
                optimizer.step()
                optimizer.zero_grad()
                loss_acc /= grad_acc
                losses.append(loss_acc)
                loss_acc = 0
                acc_n = 0
            if time_stop > 0 and time.time() - start_time > time_stop:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass


def main():
    global text, load_checkpoint, out_path, frames, rgba, save, color
    if stop_after_settings:
        print("Early exit (stop_after_settings=True)")
        return

    if capitalize:
        text = text[:1].upper() + text[1:]
    if capitalize_words:
        text = ' '.join(word[:1].upper() + word[1:] for word in text.split())  # TODO
    if capitalize_number:  # TODO
        text = ' '.join(word.upper() if word[0].isdigit() else word for word in text.split())
    print(text)
    if not same_video:
        frames = []
    try:
        frames
    except:
        frames = []
    # starting frame
    if img_path:
        Image.open(fetch(img_path)).save("source.png")
    if palette_path:
        Palette = Image.open(fetch(palette_path)).save("palette.png")
    if not same_color:
        setup()
    if display_img:
        try:
            src
        except:
            setup()
        frames = [src] * int(fps * first_for)
    try:
        color
    except:
        setup()
    if not out_path:
        out_path = text + ".mp4"
    if load_checkpoint:
        if load_gdrive:
            # from google.colab import drive
            # print("Mounting drive...")
            # drive.mount("/content/drive/")
            load_checkpoint = "/content/drive/MyDrive/" + load_checkpoint
        print("Loading checkpoint...")
        try:
            save = torch.load(load_checkpoint, map_location="cpu")
        except FileNotFoundError:
            print(f"File not found: {load_checkpoint}")
            if ignore_err:
                print("Starting training from scratch...")
            else:
                raise
        else:
            if "color" in save:
                print("Color loaded")
                color = torch.nn.Parameter(save["color"].to(device).detach())
                if "rgba" in save:
                    rgba = save["rgba"].detach()
            elif "rgba" in save:
                print("No color, training functions disabled. PLY export is available")
                rgba = save["rgba"]
                1/0
            else:
                print("No RGBA, voxel loading is available (not implemented yet)")
                print("Starting training from scratch...")
    print("First spin")
    frames += list(spin(spin_before))
    print("Training")
    if not only_spin:
        train(text=text, frames=frames)
    else:
        frames = []
    print("Last spin")
    try:
        it
        bar
    except NameError:
        it = 0
        bar = [0]
    frames += list(spin(spin_length*spin_number, total=spin_length*spin_number,
                        start=ceil(it/len(bar)*spin_length)+1,
                        spins=spin_number))
    frames += [frames[-1]] * still_frames

    # auto rename
    if os.path.exists(out_path) and rename_out:
        i = 1
        orig_path = out_path
        while os.path.exists(out_path):
            out_path = f"{orig_path.rpartition('.')[0]} ({i}).mp4"
            i += 1
    
    print("Saving video")
    p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', out_path], stdin=PIPE)
    for im in tqdm(frames):
        im.save(p.stdin, 'PNG')
    p.stdin.close()
    p.wait()
    mp4 = open(out_path, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    # display(HTML(f"""
    # <video width={w} controls>
    #       <source src="{data_url}" type="video/mp4">
    # </video>
    # """))

    # if download:
    #     from google.colab import files
    #     files.download(out_path)


if __name__ == "__main__":
    main()

#@markdown if it runs out of memory, use Runtime -> Restart Runtime


# In[ ]:


#@title convert to point cloud
# from IPython.display import clear_output, display
from tqdm.auto import trange, tqdm
#from ipywidgets import Output
import random


#@markdown note: the conversion is non-deterministic

#@markdown seed (zero if you're feeling lucky)
seed = 42 #@param {type: "integer"}
#@markdown how much to fade the shape, removing points.
#@markdown raising it increases the file size, but the quality improves.

#@markdown the file size with the old export method is 1.7 MB at fade=0.04.
fade = 0.1 #@param {type: "slider", min: 0, max: 1, step:0.01}

#@markdown display the spins every N frames
show_every = 6 #@param {type: "integer"}

#@markdown the method to turn the volume into a point cloud

#@markdown spin renders the point cloud from many views and turns it into a mesh. it's faster but uses more memory

#@markdown naïve is the method used before. its main advantage is that it's less memory hungry and more stable.

#@markdown spin only exports the surface, while naïve has the entire volume. as a result, spin is smaller but naïve might be more useful for some applications
# export_method = "spin"  #@param ["spin", "naïve"]

with torch.inference_mode():
    try:
        rgba
    except NameError:
        rgba = interpolate(color)

for export_method in ["spin"]:  # , "naïve"]:
    if not seed:
        seed = random.getrandbits(32)
    print("Fade:", fade)
    print("Seed:", seed)
    np.random.seed(seed)
    random.seed(seed)

    vertices = []

    if export_method == "spin":
        # out = Output()
        # display(out)
        
        rgbs = []
        ds = []
        try:
            for i, (rgb, depth, coords) in enumerate(spin(return_depth=True, return_coords=True)):
                rgbs.append(rgb)
                opacity = depth.sum(axis=-1)
                opacity = opacity / opacity.max()
                opaque = opacity > 0.05
                opaque = opaque * (np.random.rand(*opaque.shape) < fade)
                
                weights = np.exp(depth)
                weights = weights / weights.sum(axis=-1)[..., None]
                # zs = (weights * np.arange(0, depth.shape[-1])[None, None, :]).sum(axis=-1).astype(np.int64)
                zs = weights.argmax(axis=-1)
                ds.append(Image.fromarray(zs.astype(np.uint8)))
                # opaque = opaque * (z > 5) * (z < z.max() - 5)
                # ((h, w, 3), (h, w)) -> (h, w, 1, 3) -> (h, w, 3)
                xyzs = np.take_along_axis(coords, zs[..., None, None], axis=-2)[:, :, 0, :]
                
                yx = np.stack(np.meshgrid(*[np.arange(w) for w in xyzs.shape[:2]]), axis=-1)
                nz, = opaque.flatten().nonzero()
                yx_nz = yx.reshape((-1, 2))[nz]

                for y, x in yx_nz:
                    (x, y, z), (r, g, b) = xyzs[y, x], rgb.getpixel((int(y), int(x)))
                    if any((a < -extent or a > extent) for a in (x, y, z)):
                        continue
                    vertices.append((x, y, z, r / 255, g / 255, b / 255, 1))

                if i % show_every == 0:
                    # with out:
                    clear_output()
                    plt.hist(
                        zs.ravel()[nz],
                        bins=20)
                    plt.show_()
                    plt.axis("off")
                    plt.tight_layout()
                    plt.imshow(zs)
                    plt.colorbar()
                    plt.show_()
                    plt.axis("off")
                    plt.tight_layout()
                    plt.imshow(1 - (xyzs / 2 + 1))
                    plt.show_()
        except KeyboardInterrupt:
            pass
    elif export_method == "naïve":  # :)
        chosen = (torch.rand_like(rgba[..., 3]) < (rgba[..., 3] * fade)).long().cpu()
        xyz = torch.stack(torch.meshgrid([torch.arange(0, d) for d in chosen.shape]), dim=-1)
        
        try:
            bar = trange(rgba.shape[0])
            for x in bar:
                for y in range(rgba.shape[1]):
                    for z in range(rgba.shape[2]):
                        if not chosen[x, y, z].item():
                            continue
                        vertices.append((x / rgba.shape[0], y / rgba.shape[1], z / rgba.shape[2])
                                        + tuple(max(0, min(1, 0.5 + i.item())) for i in rgba[x, y, z]))
                bar.set_description(f"Vertices: {len(vertices)}, "
                                    f"expected: {int(len(vertices) / (x+1) * rgba.shape[0])}")
        except KeyboardInterrupt:
            pass
    print("Total:", len(vertices), "vertices.")


# In[ ]:


#@title (Only if you turn on spin) export RGB/depth videos
# from IPython.display import display, HTML
from subprocess import Popen, PIPE
from base64 import b64encode
from tqdm.auto import tqdm



def save_video(frames, out_path, fps=fps):
    p = Popen(["ffmpeg", "-y", "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-", "-vcodec", "libx264", "-r", str(fps), "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "veryslow", out_path], stdin=PIPE)
    for im in tqdm(frames):
        im.save(p.stdin, "PNG")
    p.stdin.close()
    p.wait()
    # mp4 = open(out_path, "rb").read()
    # data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    # display(HTML(f"""
    # <video width={w} controls>
    #       <source src="{data_url}" type="video/mp4">
    # </video>
    # """))


try:
    save_video(rgbs, "rgbs.mp4")
    save_video(ds, "ds.mp4")
except NameError:
    pass


# In[ ]:


#@title export to PLY
#@markdown name of the file, leave empty to name automatically
file_path = ""  #@param ["", "model.ply"] {type: "string", allow-input: true}
#@markdown path to save in google drive, leave empty if you want to store it in the colab session
drive_path = "text2voxels"  #@param ["", "text2voxels"] {type: "string", allow-input: true}
if not file_path:
    file_path = out_path.replace(".mp4", ".ply")

#@markdown download the resulting .ply file?
download = True  #@param {type: "boolean"}


def save_file(out_path):
    with open(out_path, 'w') as out_file:
        out_file.write('ply\n')
        out_file.write('format ascii 1.0\n')
        out_file.write(f'element vertex {len(vertices)}\n') # variable # of vertices
        out_file.write('property float x\n')
        out_file.write('property float y\n')
        out_file.write('property float z\n')
        out_file.write('property uchar red\n')
        out_file.write('property uchar green\n')
        out_file.write('property uchar blue\n')
        out_file.write('property uchar alpha\n')
        out_file.write('end_header\n')
        for x, y, z, r, g, b, a in tqdm(vertices):
            out_file.write(f"{x} {rgba.shape[1] - y} {z} {int(r * 255)} {int(g * 255)} {int(b * 255)} {int(a * 255)}\n")


save_file(file_path)

# if download:
#     from google.colab import files
#     files.download(file_path)
#
# if drive_path:
#     from google.colab import drive
#     print("Mounting drive...")
#     drive.mount("/content/drive/")
#     print("Saving to drive...")
#     drive_path = f"/content/drive/MyDrive/{drive_path}"
#     os.makedirs(drive_path, exist_ok=True)
#     save_file(f"{drive_path}/{file_path}")

#@markdown then open this in meshlab and enjoy!


# In[ ]:


# get_ipython().run_cell_magic('time', '', '#@title (optional, experimental) saving the intermediate representation\n#@markdown this is the old export option, but these results might be useful in the future when volume export is added.\n\n#@markdown you can also continue an interrupted generation using this\n\n#@markdown disabled by default, check `save_color` to enable\n\n#@markdown warning: the files take up about 50MB compressed at 128x128x128.\n\n#@markdown name of the file, leave empty to name automatically\nfile_path = ""  #@param ["", "model.pth"] {type: "string", allow-input: true}\n#@markdown path to save in google drive, leave empty if you want to store it in the colab session\ndrive_path = ""  #@param ["", "text2voxels"] {type: "string", allow-input: true}\nif not file_path:\n    file_path = out_path.replace(".mp4", ".pth")\n#@markdown which parts of the model to save.\n\n#@markdown with color it\'s possible to restore the model, but you won\'t be able\n#@markdown to open it outside the notebook. with rgba it might be more portable\n#@markdown but you won\'t be able to restore and train for longer\nsave_color = False #@param {type: "boolean"}\nsave_rgba = True #@param {type: "boolean"}\n# RGB saving removed\nsave_rgb = False  #@#param {type: "boolean"}\n\n#@markdown download the resulting .pth file?\ndownload = True  #@param {type: "boolean"}\n\nwith torch.inference_mode():\n    try:\n        rgba\n    except NameError:\n        rgba = interpolate(color)\n\n    rgb = rgba[..., :3] * rgba[..., -1:]\nto_save = {}\nif save_color:\n    to_save.update(dict(\n        color=color,\n        scale_from=scale_from,\n        scale_decay=scale_decay,\n        grayscale=grayscale\n    ))\nif save_rgba:\n    to_save["rgba"] = rgba\nif save_rgb:\n    to_save["rgb"] = rgb\nprint("Saving...")\ntorch.save(to_save, file_path)\n\nif download:\n    from google.colab import files\n    files.download(file_path)\n\nif drive_path:\n    from google.colab import drive\n    print("Mounting drive...")\n    drive.mount("/content/drive/")\n    print("Saving to drive...")\n    drive_path = f"/content/drive/MyDrive/{drive_path}"\n    os.makedirs(drive_path, exist_ok=True)\n    torch.save(to_save, f"{drive_path}/{file_path}")\n')


# the end

# neural_style_transfer_app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Functions
def load_image(uploaded_file, max_size=400):
    image = Image.open(uploaded_file).convert('RGB')
    size = max(image.size) if max(image.size) < max_size else max_size
    
    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image.to(device)

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # unnormalize
    image = np.clip(image, 0, 1)
    return image

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',  
            '5': 'conv2_1',  
            '10': 'conv3_1', 
            '19': 'conv4_1', 
            '21': 'conv4_2', # content representation
            '28': 'conv5_1'  
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def style_transfer(content, style, steps=500, style_weight=1e2, content_weight=1e4):
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    target = content.clone().requires_grad_(True).to(device)

    optimizer = optim.Adam([target], lr=0.003)

    style_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.75,
        'conv3_1': 0.2,
        'conv4_1': 0.2,
        'conv5_1': 0.2
    }

    for ii in range(1, steps+1):
        target_features = get_features(target, vgg)

        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            b, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    return target

# Streamlit App
st.title("🎨 Neural Style Transfer App")
st.write("Upload a **content image** (your photo) and a **style image** (a painting), and watch the magic happen!")

content_file = st.file_uploader("Upload Content Image", type=['jpg', 'jpeg', 'png'])
style_file = st.file_uploader("Upload Style Image", type=['jpg', 'jpeg', 'png'])

if content_file and style_file:
    content = load_image(content_file)
    style = load_image(style_file)
    
    st.image(Image.open(content_file), caption='Content Image', width=300)
    st.image(Image.open(style_file), caption='Style Image', width=300)
    
    if st.button('Stylize!'):
        with st.spinner('Stylizing... This might take a while ⏳'):
            output = style_transfer(content, style)
            output_image = im_convert(output)
            st.image(output_image, caption='Stylized Image', use_column_width=True)
            st.success('Done!')


import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms
from timm.models.pvt import pvt_small
from timm.models.resnet import resnet50d
from timm.models.ccmlpv3 import ccmlp_b1, ccmlp_b0_baseline
from timm.models.mlp_mixer import mixer_b16_224
from timm.models import create_model, resume_checkpoint, convert_splitbn_model
import pdb

import argparse
import os

parser = argparse.ArgumentParser(description='PyTorch Visualize')
parser.add_argument('--model', metavar='MODEL', default='vit',
                    help='path to dataset')
def main_pvt():
    #pdb.set_trace()
    model = create_model(
            'pvt_small',
            checkpoint_path='pretrain/pvt_small.pth',
            pretrained=True,
            num_classes=1000)

    block_weights = [] # we will save the conv layer weights in this list
    block_layers = [] # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())

    all_block = []  # put all sub block in the list
    for model_child in list(model.children()):
        if type(model_child) != torch.nn.modules.container.ModuleList:
            all_block.append(model_child)
        else:
            sub_children = list(model_child.children())
            for sub_child in sub_children:
                all_block.append(sub_child)
    pdb.set_trace()
    # read and visualize an image
    img = cv2.imread("images/dog.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384,384)),  # reszie the image as (1024, 1024) or (2048, 2048) to obtain high resolution features if you have large CPU or GPU memory
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())


    # pass the image through all the layers
    results = []
    results = [all_block[0](img)]
    for i in range(1, len(all_block)-2):
        # pass the result from the last layer to the next layer
        #print(i)
        results.append(all_block[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    all_layers_num = 5   # change the number of length to control how many layers you want to visualize
    #pdb.set_trace()
    for num_layer in range(all_layers_num): 
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer]#[0, :, :, :]
        layer_viz = layer_viz.data
        layer_viz = layer_viz.squeeze(0).transpose(0, 1)
        layer_viz = layer_viz.reshape(1024, 24, 24)  # chage the reshape here
        #print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 36: # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(6, 6, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"output/vis/swin/layer_{num_layer}.png")  # change the path to save the feature maps
        #plt.show() # use this line to show the figure in jupyter notebook
        plt.close()

def main_resnet():
    model = create_model(
            'resnet50d',
            checkpoint_path='pretrain/resnet50d.pth', 
            pretrained=False,
            num_classes=1000)

    block_weights = [] # we will save the conv layer weights in this list
    block_layers = [] # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())

    all_block = []  # put all sub block in the list
    for model_child in list(model.children()):
        if type(model_child) != torch.nn.modules.container.ModuleList:
            all_block.append(model_child)
        else:
            sub_children = list(model_child.children())
            for sub_child in sub_children:
                all_block.append(sub_child)
    #pdb.set_trace()
    # read and visualize an image
    img = cv2.imread("images/dog.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024,1024)),  # reszie the image as (1024, 1024) or (2048, 2048) to obtain high resolution features if you have large CPU or GPU memory
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    #pdb.set_trace()
    # pass the image through all the layers
    results = []
    layers = [0,4,5,6,7]
    #results = [all_block[0](img)]
    for i in range(len(all_block)):
        # pass the result from the last layer to the next layer
        img = all_block[i](img)
        if i in layers:
            results.append(img)
    #outputs = results

    stages = 5   # change the number of length to control how many layers you want to visualize
    #pdb.set_trace()
    for num_stage in range(stages): 
        plt.figure(figsize=(16, 12))
        layer_viz = results[num_stage]#[0, :, :, :]
        layer_viz = layer_viz.data
        layer_viz = layer_viz.squeeze(0)
        #layer_viz = layer_viz.squeeze(0).transpose(0, 1)
        #layer_viz = layer_viz.reshape(1024, 24, 24)  # chage the reshape here
        #print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 12: # we will visualize only 3x6 blocks from each layer
                break
            plt.subplot(3, 4, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_stage} feature maps...")
        plt.savefig(f"output/vis/resnet/stage_{num_stage}.png")  # change the path to save the feature maps
        #plt.show() # use this line to show the figure in jupyter notebook
        plt.close()

def main_convmlp():
    model = create_model(
            'ccmlp_b1',
            checkpoint_path='pretrain/ccmlp_b1.pth', 
            pretrained=False,
            num_classes=1000)

    block_weights = [] # we will save the conv layer weights in this list
    block_layers = [] # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())

    all_block = []  # put all sub block in the list
    for model_child in list(model.children()):
        if type(model_child) != torch.nn.modules.container.ModuleList:
            all_block.append(model_child)
        else:
            sub_children = list(model_child.children())
            for sub_child in sub_children:
                all_block.append(sub_child)
    #pdb.set_trace()
    # read and visualize an image
    img = cv2.imread("images/dog.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024,1024)),  # reszie the image as (1024, 1024) or (2048, 2048) to obtain high resolution features if you have large CPU or GPU memory
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    #pdb.set_trace()
    # pass the image through all the layers
    results = []
    layers = [0,1,2,3,4]
    #results = [all_block[0](img)]
    for i in range(5):
        # pass the result from the last layer to the next layer
        if i == 2:
            img = img.permute(0, 2, 3, 1)
        img = all_block[i](img)
        if i in layers:
            results.append(img)
    #outputs = results

    stages = 5   # change the number of length to control how many layers you want to visualize
    #pdb.set_trace()
    for num_stage in range(stages): 
        plt.figure(figsize=(16, 12))
        layer_viz = results[num_stage]#[0, :, :, :]
        layer_viz = layer_viz.data
        layer_viz = layer_viz.squeeze(0)
        if num_stage > 1:
            layer_viz = layer_viz.permute(2,0,1)
        #pdb.set_trace()
        #layer_viz = layer_viz.squeeze(0).transpose(0, 1)
        #layer_viz = layer_viz.reshape(1024, 24, 24)  # chage the reshape here
        #print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 12: # we will visualize only 6x6 blocks from each layer
                break
            plt.subplot(3, 4, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_stage} feature maps...")
        plt.savefig(f"output/vis/convmlp/stage_{num_stage}.png")  # change the path to save the feature maps
        #plt.show() # use this line to show the figure in jupyter notebook
        plt.close()

def main_basemlp():
    model = create_model(
            'ccmlp_b0_baseline',
            checkpoint_path='pretrain/ccmlp_b0_baseline.pth', 
            pretrained=False,
            num_classes=1000)

    block_weights = [] # we will save the conv layer weights in this list
    block_layers = [] # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())

    all_block = []  # put all sub block in the list
    for model_child in list(model.children()):
        if type(model_child) != torch.nn.modules.container.ModuleList:
            all_block.append(model_child)
        else:
            sub_children = list(model_child.children())
            for sub_child in sub_children:
                all_block.append(sub_child)
    #pdb.set_trace()
    # read and visualize an image
    img = cv2.imread("images/dog.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024,1024)),  # reszie the image as (1024, 1024) or (2048, 2048) to obtain high resolution features if you have large CPU or GPU memory
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    #pdb.set_trace()
    # pass the image through all the layers
    results = []
    layers = [0,1,2,3,4]
    #results = [all_block[0](img)]
    for i in range(5):
        # pass the result from the last layer to the next layer
        if i == 1:
            img = img.permute(0, 2, 3, 1)
        img = all_block[i](img)
        if i in layers:
            results.append(img)
    #outputs = results

    stages = 5   # change the number of length to control how many layers you want to visualize
    #pdb.set_trace()
    for num_stage in range(stages): 
        plt.figure(figsize=(16, 12))
        layer_viz = results[num_stage]#[0, :, :, :]
        layer_viz = layer_viz.data
        layer_viz = layer_viz.squeeze(0)
        if num_stage > 0:
            layer_viz = layer_viz.permute(2,0,1)
        #pdb.set_trace()
        #layer_viz = layer_viz.squeeze(0).transpose(0, 1)
        #layer_viz = layer_viz.reshape(1024, 24, 24)  # chage the reshape here
        #print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 12: # we will visualize only 6x6 blocks from each layer
                break
            plt.subplot(3, 4, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_stage} feature maps...")
        plt.savefig(f"output/vis/mlp-base/stage_{num_stage}.png")  # change the path to save the feature maps
        #plt.show() # use this line to show the figure in jupyter notebook
        plt.close()

def main_mixer():
    model = create_model(
            'mixer_b16_224',
            #checkpoint_path='pretrain/mixer_b16_224.pth', 
            pretrained=True,
            num_classes=1000)

    block_weights = [] # we will save the conv layer weights in this list
    block_layers = [] # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())

    all_block = []  # put all sub block in the list
    for model_child in list(model.children()):
        #pdb.set_trace()
        #if type(model_child) != torch.nn.modules.container.ModuleList:
        #    all_block.append(model_child)
        #else:
        sub_children = list(model_child.children())
        for sub_child in sub_children:
            all_block.append(sub_child)
    #pdb.set_trace()
    # read and visualize an image
    img = cv2.imread("images/dog.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),  # reszie the image as (1024, 1024) or (2048, 2048) to obtain high resolution features if you have large CPU or GPU memory
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())

    #pdb.set_trace()
    # pass the image through all the layers
    results = []
    layers = [2,5,8,11]
    #results = [all_block[0](img)]
    for i in range(12):
        # pass the result from the last layer to the next layer
        if i == 2:
            img = img.flatten(2).transpose(1, 2)
        img = all_block[i](img)
        if i in layers:
            results.append(img)
    #outputs = results

    stages = 4   # change the number of length to control how many layers you want to visualize
    #pdb.set_trace()
    for num_stage in range(stages): 
        plt.figure(figsize=(16, 12))
        layer_viz = results[num_stage]#[0, :, :, :]
        layer_viz = layer_viz.data
        layer_viz = layer_viz.squeeze(0).transpose(0, 1).reshape(768, 14, 14)
        #pdb.set_trace()
        #if num_stage > 0:
        #    layer_viz = layer_viz.permute(2,0,1)
        #pdb.set_trace()
        #layer_viz = layer_viz.squeeze(0).transpose(0, 1)
        #layer_viz = layer_viz.reshape(1024, 24, 24)  # chage the reshape here
        #print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 12: # we will visualize only 6x6 blocks from each layer
                break
            plt.subplot(3, 4, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_stage} feature maps...")
        plt.savefig(f"output/vis/mixer/stage_{num_stage}.png")  # change the path to save the feature maps
        #plt.show() # use this line to show the figure in jupyter notebook
        plt.close()
    
if __name__ == '__main__':
    args = parser.parse_args()
    if args.model == 'pvt':
        main_pvt()
    elif args.model == 'resnet':
        main_resnet()
    elif args.model == 'convmlp':
        main_convmlp()
    elif args.model == 'basemlp':
        main_basemlp()
    elif args.model == 'mixer':
        main_mixer()
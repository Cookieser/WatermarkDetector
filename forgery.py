import torch
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from stable_signature_util import *


def Forgery_no_box(original_image, watermarked_image, key, criterion, decoder,clip,args):

    
    r = args.r
    lr = args.lr
    epsilon = args.epsilon
    iteration = args.iteration
    original_image = original_image.cuda()
    watermarked_image =  watermarked_image.cuda()
    original_image_cloned = original_image.clone()

    success = False
    acc_list = []
    diff_list = []


    for i in range(iteration):
        original_image = original_image.requires_grad_(True)
        min_value, max_value = torch.min(original_image), torch.max(original_image)

        # Post-process the watermarked image.
        original_image_feature = clip.encode(original_image, no_grad=False)
        watermarked_image_feature = clip.encode( watermarked_image, no_grad=False)

        loss = criterion(original_image_feature, watermarked_image_feature)

        grads = torch.autograd.grad(loss, original_image)
        with torch.no_grad():
            original_image = original_image - lr * grads[0]
            original_image = torch.clamp(original_image, min_value, max_value)

        # Projection.
        perturbation_norm = torch.norm(original_image - original_image_cloned, float('inf'))
        if perturbation_norm.cpu().detach().numpy() >= r:
            c = r / perturbation_norm
            original_image = project(original_image, original_image_cloned, c)

        mes = watermark_decoder(decoder,original_image)
        acc = computer_acc(mes,key)


        acc_list.append(acc)
        diff_list.append(loss.item())
        print(f"iteration:{i}           diff:{loss.item()}     bit_acc_target:{acc}")
        
        # Early Stopping.
        if perturbation_norm.cpu().detach().numpy() >= r:
            print("stop because of the perturbation_norm....")
            break

        if acc >= 1 - epsilon:
            success = True
            break
        
        bound = torch.norm(original_image - original_image_cloned, float('inf'))


    return acc, bound.item(), success, original_image,acc_list,diff_list

def Forgery_no_box_classifier(original_image, key, criterion, decoder,classifier,args):

    
    r = args.r
    lr = args.lr
    epsilon = args.epsilon
    iteration = args.iteration


    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)
    original_image = original_image.cuda()
    original_image_cloned = original_image.clone()

    target_label = torch.ones(original_image.size(0), dtype=torch.long,device = original_image.device)
    acc_list, diff_list = [], []
    success = False


    


    for i in range(iteration):
        original_image = original_image.requires_grad_(True)
        min_value, max_value = torch.min(original_image), torch.max(original_image)

        # Post-process the watermarked image.
        logits = classifier(original_image)
        loss   = criterion(logits, target_label)
        preds  = logits.argmax(dim=1)
        acc    = (preds == target_label).float().mean().item()

        grads = torch.autograd.grad(loss, original_image)
        with torch.no_grad():
            original_image = original_image - lr * grads[0]
            original_image = torch.clamp(original_image, min_value, max_value)

        # Projection.
        perturbation_norm = torch.norm(original_image - original_image_cloned, float('inf'))
        if perturbation_norm.cpu().detach().numpy() >= r:
            c = r / perturbation_norm
            original_image = project(original_image, original_image_cloned, c)


        mes = watermark_decoder(decoder,original_image)
        watermark_acc = computer_acc(mes,key)
        
        bound = torch.max(torch.abs(original_image - original_image_cloned)).item()
        acc_list.append(acc)
        diff_list.append(loss.item())              
        label_map = {0: "Original", 1: "Watermarked"}
        print(f"iter:{i:<3}  loss:{loss.item():.4f}  Detect: {label_map[acc]} watermark_bit_wise_acc:{watermark_acc:.6f}")


        
        # Early Stopping.
        if perturbation_norm.cpu().detach().numpy() >= r:
            print("stop because of the perturbation_norm....")
            break

        # if acc >= 1 - epsilon:
        #     print(f"acc:{acc}")
        #     success = True
        #     break
        
        bound = perturbation_norm.item()


    return acc, bound, success, original_image,original_image_cloned, acc_list,diff_list


def transform_image(image):
    # For HiDDeN watermarking method, image pixel value range should be [-1, 1]. Transform an image into [-1, 1] range.
    cloned_encoded_images = (image + 1) / 2  # for HiDDeN watermarking method only
    cloned_encoded_images = cloned_encoded_images.mul(255).clamp_(0, 255)

    cloned_encoded_images = cloned_encoded_images / 255
    cloned_encoded_images = cloned_encoded_images * 2 - 1  # for HiDDeN watermarking method only
    image = cloned_encoded_images.cuda()

    return image

def project(param_data, backup, epsilon):
    # If the perturbation exceeds the upper bound, project it back.
    r = param_data - backup
    r = epsilon * r
    return backup + r




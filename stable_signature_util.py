
from PIL import Image
import torch
import torchvision.transforms as transforms


def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]



def watermark_decoder(msg_extractor,img):

    transform_imnet = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])


    img = transform_imnet(img).to("cuda")

    msg = msg_extractor(img) # b c h w -> b k
    # print("Logits: ", msg.squeeze().cpu().detach().numpy())
    bool_msg = (msg>0).squeeze().cpu().numpy().tolist()
    # print("Extracted message: ", msg2str(bool_msg))
    return bool_msg


def computer_acc(bool_msg,key):
    
    bool_key = str2msg(key)
    # compute difference between model key and message extracted from image
    diff = [bool_msg[i] != bool_key[i] for i in range(len(bool_msg))]
    bit_acc = 1 - sum(diff)/len(diff)
    # print("Bit accuracy: ", bit_acc)
    return bit_acc
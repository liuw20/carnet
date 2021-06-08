from torch.autograd import Variable
import numpy as np
from utils.dataloader import selfData, collate_fn
import torch
import torch.nn as nn
from model.malexnet import mAlexNet
from model.alexnet import AlexNet
from model.carnet import carNet
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from utils.options import args_parser
from tqdm import tqdm
import torchvision.models as models
import model.shuffleNetV2SE
import model.shuffleNetV2SE
import model.shuffleNet
import model.shuffleNet_K5
import model.shufflenet_v2_liteconv
import model.shufflenet_v2_k5_liteconv
import model.shufflenet_v2_csp
import model.shufflenet_v2_sk_attention

args = args_parser()
transforms = transforms.Compose([
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Resize((224,224)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
def test(img_path,target_path, transforms, net):
    print("\nTesting starts now...")
    net.eval()
    test_dataset = selfData(img_path, target_path, transforms)
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 8,drop_last= False,pin_memory=True,collate_fn=collate_fn)
    # test_loader=list(test_loader)
    test_size=len(test_dataset)
    data_size=test_size//64
    if test_size%64 !=0:
        data_size+=1
    correct = 0
    total = 0
    item = 1
    test_iter=iter(test_loader)
    with torch.no_grad():
        for i in tqdm(range(data_size)):
        # for data in test_loader:
        #     data=next(test_iter)
        #     images, labels = data
            images,labels=next(test_iter)
            # print("Testing on batch {}".format(item))
            labels = list(map(int, labels))
            labels = torch.Tensor(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            item += 1
    return (correct/total)

# def eval(model_root,img_path,target_path, transforms, net):
#     net.load_state_dict(torch.load(model_root))
#     test_data = selfData(img_path, target_path, transforms)
#     data_size=len(test_data)
#     net.eval()
#
#     correct=0
#     total=0
#     with torch.no_grad():
#         # for i in tqdm(range(data_size)):
#         for i in range(data_size):
#             images,labels=test_data[i]
#             if images is None:
#                 continue
#             images=Variable(images.unsqueeze(0))
#             # print(images.shape)
#             # exit()
#             # labels=Variable(labels.unsqueeze(0))
#             # labels = list(map(int, labels))
#             # labels = torch.Tensor(labels)
#             if torch.cuda.is_available():
#                 device = torch.device(args.cuda_device)
#                 images = images.to(device)
#                 # labels = labels.to(device)
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += 1
#             print(predicted,labels)
#             if predicted==labels:
#                 print("yes")
#                 correct+=1
#             # correct += (predicted == labels)
#             # exit()
#         print("Acc:",(correct/total))

if __name__=="__main__":
    if args.model=='mobilenet_v3_small':
        net = models.mobilenet_v3_small()
        net.classifier[3] = nn.Linear(1024, 2)
    elif args.model=='mobilenet_v2':
        net = models.mobilenet_v2()
        net.classifier[1] = nn.Linear(1280, 2)
    elif args.model=="ShuffleNet_v2":
        net=models.shufflenet_v2_x1_0()
        net.fc = nn.Linear(1024, 2)
    elif args.model == "mobilenet_v3_large":
        net = models.mobilenet_v3_large()
        net.classifier[3] = nn.Linear(1280, 2)
    elif args.model == 'mAlexNet':
        net = mAlexNet()
    elif args.model=='carnet':
        net=carNet()
    elif args.model == "ShuffleNet_v2_SE":
        net = model.shuffleNetV2SE.ShuffleNetV2SE()
    elif args.model == "ShuffleNet_v2":
        net = model.shuffleNet.ShuffleNetV2()
    elif args.model == "ShuffleNet_K5":
        net = model.shuffleNet_K5.ShuffleNetV2K5()
    elif args.model == 'ShuffleNet_v2_csp':
        net = model.shufflenet_v2_csp.ShuffleNetV2CSP()
    elif args.model == 'ShuffleNet_v2_k5_liteconv':
        net = model.shufflenet_v2_k5_liteconv.ShuffleNetV2K5Lite()
    elif args.model == 'ShuffleNet_v2_sk':
        net = model.shufflenet_v2_sk_attention.ShuffleNetV2SK()
    elif args.model == 'ShuffleNet_v2_liteconv':
        net = model.shufflenet_v2_liteconv.ShuffleNetV2LiteConv()
    torch.set_default_tensor_type('torch.FloatTensor')
    print("test net:{}..".format(args.model))
    # print({k.replace('module.',''):v for k,v in torch.load(args.path,map_location="cpu").items()})
    # exit()
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.path,map_location="cpu").items()})
    # exit()
    if torch.cuda.is_available():
        net.cuda()
    acc=test(args.test_img,args.test_lab,transforms,net)
    print(args.test_lab,acc)
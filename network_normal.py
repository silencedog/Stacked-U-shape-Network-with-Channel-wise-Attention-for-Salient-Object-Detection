import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_dilation import resnet50
from setting import _ASPPModule, crop
from attention import CAM_Module
from pdcm import PDCM
from collections import OrderedDict


class CAMUNet(nn.Module):
    def __init__(self, input_channels=3, pretrained=True):
        super(CAMUNet, self).__init__()
        self.resnet = resnet50(pretrained=pretrained, input_channels=input_channels)
        self.aspp   = _ASPPModule(2048, 256)
        self.conv   =  nn.Conv2d(256, 1, (1, 1), stride=1)
        self.pdcmblock0 = PDCM(64)
        self.pdcmblock1 = PDCM(256)
        self.pdcmblock2 = PDCM(512)
        self.pdcmblock3 = PDCM(1024)
        self.pdcmblock4 = PDCM(256)
        self.upsample_2_2 = nn.ConvTranspose2d(64, 64, 4, stride=2, bias=False)
        self.upsample_2_3 = nn.ConvTranspose2d(64, 64, 4, stride=2, bias=False)
        self.upsample_2_4 = nn.ConvTranspose2d(64, 64, 4, stride=2, bias=False)
        self.upsample_2_5 = nn.ConvTranspose2d(32, 32, 4, stride=2, bias=False)
        
        self.cam_conv1_1  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(64)),
            ('relu', nn.PReLU(num_parameters=64))]))
        self.cam1         = CAM_Module(64)
        self.cam_conv1_2  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(64)),
            ('relu', nn.PReLU(num_parameters=64))]))
        self.cam_drop1    = nn.Dropout2d(0.1, False)
        self.cam_conv1_3  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, 32, 1, stride=1)),
            ('bn'  , nn.BatchNorm2d(32)),
            ('relu', nn.PReLU(num_parameters=32))]))
        self.cam_down_c1  =  nn.Conv2d(32, 1, 1, stride=1)
        
        self.cam_conv2_1  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(320, 128, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(128)),
            ('relu', nn.PReLU(num_parameters=128))]))
        self.cam2         = CAM_Module(128)
        self.cam_conv2_2  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(64)),
            ('relu', nn.PReLU(num_parameters=64))]))
        self.cam_drop2    = nn.Dropout2d(0.1, False)
        self.cam_conv2_3  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, 64, 1, stride=1)),
            ('bn'  , nn.BatchNorm2d(64)),
            ('relu', nn.PReLU(num_parameters=64))]))
        self.cam_down_c2  =  nn.Conv2d(64, 1, 1, stride=1)
        
        self.cam_conv3_1  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(576, 256, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(256)),
            ('relu', nn.PReLU(num_parameters=256))]))
        self.cam3         = CAM_Module(256)
        self.cam_conv3_2  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(256, 128, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(128)),
            ('relu', nn.PReLU(num_parameters=128))]))
        self.cam_drop3    = nn.Dropout2d(0.1, False)
        self.cam_conv3_3  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(128, 64, 1, stride=1)),
            ('bn'  , nn.BatchNorm2d(64)),
            ('relu', nn.PReLU(num_parameters=64))]))
        self.cam_down_c3  =  nn.Conv2d(64, 1, 1, stride=1)

        self.cam_conv4_1  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(1056, 512, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(512)),
            ('relu', nn.PReLU(num_parameters=512))]))
        self.cam4         = CAM_Module(512)
        self.cam_conv4_2  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(512, 256, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(256)),
            ('relu', nn.PReLU(num_parameters=256))]))
        self.cam_drop4    = nn.Dropout2d(0.1, False)
        self.cam_conv4_3  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(256, 64, 1, stride=1)),
            ('bn'  , nn.BatchNorm2d(64)),
            ('relu', nn.PReLU(num_parameters=64))]))
        self.cam_down_c4  =  nn.Conv2d(64, 1, 1, stride=1)
        
        self.cam_conv5_1  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(256, 128, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(128)),
            ('relu', nn.PReLU(num_parameters=128))]))
        self.cam5         = CAM_Module(128)
        self.cam_conv5_2  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(128, 64, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(64)),
            ('relu', nn.PReLU(num_parameters=64))]))
        self.cam_drop5    = nn.Dropout2d(0.1, False)
        self.cam_conv5_3  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, 32, 1, stride=1)),
            ('bn'  , nn.BatchNorm2d(32)),
            ('relu', nn.PReLU(num_parameters=32))]))
        self.cam_down_c5  =  nn.Conv2d(32, 1, 1, stride=1)

        self.conv_cat_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 32, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(32)),
            ('relu', nn.PReLU(num_parameters=32))]))
        self.conv_cat_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, 16, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(16)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_cat_3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, 16, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(16)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_cat_4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(64, 16, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(16)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_cat_5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 16, 3, stride=1, padding=1)),
            ('bn'  , nn.BatchNorm2d(16)),
            ('relu', nn.PReLU(num_parameters=16))]))

        self.conv_p1_0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 32, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=32))]))
        self.conv_p1_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(32, 32, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=32))]))
        self.conv_p1_2 = nn.Conv2d(32, 1, 1, stride=1)

        self.conv_p2_up_0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p2_up_1 = nn.Conv2d(16, 1, 1, stride=1)

        self.conv_p3_up_0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p3_up_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p3_up_2 = nn.Conv2d(16, 1, 1, stride=1)

        self.conv_p4_up_0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p4_up_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p4_up_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p4_up_3 = nn.Conv2d(16, 1, 1, stride=1)

        self.conv_p5_up_0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p5_up_1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p5_up_2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 16, 3, stride=1, padding=1)),
            ('relu', nn.PReLU(num_parameters=16))]))
        self.conv_p5_up_3 = nn.Conv2d(16, 1, 1, stride=1)

        self.fuse         =  nn.Conv2d(36, 1, 1, stride=1)

        if pretrained:                                                         
            for key in self.state_dict():
                if 'resnet' not in key:
                    self.init_layer(key)

    def init_layer(self, key):
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                if self.state_dict()[key].ndimension() >= 2:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out', nonlinearity='relu')
            elif 'bn' in key:
                self.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            self.state_dict()[key][...] = 0.001

    def feat_conv(self, x):
        '''
            Spatial feature extractor
        '''
        block0 = self.resnet.conv1(x)
        block0 = self.resnet.bn1(block0)
        block0 = self.resnet.relu(block0)
        block0 = self.resnet.maxpool(block0)

        block1 = self.resnet.layer1(block0)
        block2 = self.resnet.layer2(block1)
        block3 = self.resnet.layer3(block2)
        block4 = self.resnet.layer4(block3)
        block4 = self.aspp(block4)
        return block0, block1, block2, block3, block4

  
    def forward(self, x):
        all_fuse = list()
        block0, block1, block2, block3, block4 = self.feat_conv(x)

        block0 = self.pdcmblock0(block0)
        block1 = self.pdcmblock1(block1)
        block2 = self.pdcmblock2(block2)
        block3 = self.pdcmblock3(block3)
        block4 = self.pdcmblock4(block4)

        cam_feat5   = self.cam_conv5_1(block4)
        cam_feat5   = self.cam5(cam_feat5)
        cam_conv5   = self.cam_conv5_2(cam_feat5)
        cam_conv5   = self.cam_drop5(cam_conv5) 
        cam_output5 = self.cam_conv5_3(cam_conv5)

        concat4     = torch.cat([block3, cam_output5], 1)
        
        cam_feat4   = self.cam_conv4_1(concat4)
        cam_feat4   = self.cam4(cam_feat4)
        cam_conv4   = self.cam_conv4_2(cam_feat4)
        cam_conv4   = self.cam_drop4(cam_conv4) 
        cam_output4 = self.cam_conv4_3(cam_conv4)
        cam_output4 = self.upsample_2_4(cam_output4)
        cam_output4 = crop(cam_output4, x, 2, 2)
        
        concat3     = torch.cat([block2, cam_output4], 1)
        
        cam_feat3   = self.cam_conv3_1(concat3)
        cam_feat3   = self.cam3(cam_feat3)
        cam_conv3   = self.cam_conv3_2(cam_feat3)
        cam_conv3   = self.cam_drop3(cam_conv3) 
        cam_output3 = self.cam_conv3_3(cam_conv3)
        cam_output3 = self.upsample_2_3(cam_output3)
        cam_output3 = crop(cam_output3, x, 2, 2)
        
        concat2     = torch.cat([block1, cam_output3], 1)
        
        cam_feat2   = self.cam_conv2_1(concat2)
        cam_feat2   = self.cam2(cam_feat2)
        cam_conv2   = self.cam_conv2_2(cam_feat2)
        cam_conv2   = self.cam_drop2(cam_conv2) 
        cam_output2 = self.cam_conv2_3(cam_conv2)
        
        concat1     = torch.cat([block0, cam_output2], 1)
        
        cam_feat1   = self.cam_conv1_1(concat1)
        cam_feat    = cam_feat1
        cam_feat1   = self.cam1(cam_feat1)
        cam_conv1   = self.cam_conv1_2(cam_feat1)
        cam_conv1   = self.cam_drop1(cam_conv1)
        cam_output1 = self.cam_conv1_3(cam_conv1)

        l12   = self.conv_cat_1(cam_output1)
        l22   = self.conv_cat_2(cam_output2)
        l32   = self.conv_cat_3(cam_output3)
        l42   = self.conv_cat_4(cam_output4)
        l52   = self.conv_cat_5(cam_output5)

        p5    = l52
        p5_up = nn.functional.interpolate(self.conv_p5_up_0(p5), 32, mode='bilinear', align_corners=False)
        p5_up = nn.functional.interpolate(self.conv_p5_up_1(p5_up), 64, mode='bilinear', align_corners=False)
        p5_up = nn.functional.interpolate(self.conv_p5_up_2(p5_up), 128, mode='bilinear', align_corners=False)
        p5_up = nn.functional.interpolate(self.conv_p5_up_3(p5_up), 256, mode='bilinear', align_corners=False)

        p4    = l42
        p4_up = nn.functional.interpolate(self.conv_p4_up_1(p4), 64, mode='bilinear',align_corners=False)
        p4_up = nn.functional.interpolate(self.conv_p4_up_2(p4_up), 128, mode='bilinear',align_corners=False)
        p4_up = nn.functional.interpolate(self.conv_p4_up_3(p4_up), 256, mode='bilinear',align_corners=False)

        p3    = l32
        p3_up = nn.functional.interpolate(self.conv_p3_up_1(p3), 128, mode='bilinear',align_corners=False)
        p3_up = nn.functional.interpolate(self.conv_p3_up_2(p3_up), 256, mode='bilinear', align_corners=False)

        p2    =  l22
        p2_up = nn.functional.interpolate(self.conv_p2_up_0(p2), 128, mode='bilinear', align_corners=False)
        p2_up = nn.functional.interpolate(self.conv_p2_up_1(p2_up), 256, mode='bilinear', align_corners=False)

        p1    =  l12
        p1_up = nn.functional.interpolate(self.conv_p1_0(p1) , 128, mode='bilinear', align_corners=False)
        p1_up = nn.functional.interpolate(self.conv_p1_1(p1_up), 256, mode='bilinear', align_corners=False)
        p1_up_1 = self.conv_p1_2(p1_up)
        fuse  = self.fuse(torch.cat([p1_up, p2_up, p3_up, p4_up, p5_up],1))


        fuse         = torch.sigmoid(fuse)
        p1_up_1      = torch.sigmoid(p1_up_1)
        p2_up        = torch.sigmoid(p2_up)
        p3_up        = torch.sigmoid(p3_up)
        p4_up        = torch.sigmoid(p4_up)
        p5_up        = torch.sigmoid(p5_up)

        

        all_fuse.append(fuse)
        all_fuse.append(p1_up_1)
        all_fuse.append(p2_up)
        all_fuse.append(p3_up)
        all_fuse.append(p4_up)
        all_fuse.append(p5_up)

        return all_fuse

def build_model():
    return CAMUNet()


if __name__ == '__main__':

    net = build_model()
    img = torch.rand((4, 3, 256, 256))  # output is (1,256,256)!
    net = net.to(torch.device('cuda:0'))
    img = img.to(torch.device('cuda:0'))
    pre_feat = torch.rand((4, 256, 16, 16)).cuda()
    out = net(img)
    #for x in net(img):
       # print(x.data.shape)
    print(net)


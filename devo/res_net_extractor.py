import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer import BasicBlock, conv1x1, conv3x3

class ResNetFPN(nn.Module):
    """
    ResNet18, output resolution is 1/8.
    Each block has 2 layers.
    """
    def __init__(self, args, input_dim=3, output_dim=256, ratio=1.0, norm_layer=nn.BatchNorm2d, init_weight=False):
        super().__init__()
        # Config
        block = BasicBlock
        block_dims = args.block_dims
        initial_dim = args.initial_dim
        self.init_weight = init_weight
        self.input_dim = input_dim
        # Class Variable
        self.in_planes = initial_dim
        for i in range(len(block_dims)):
            block_dims[i] = int(block_dims[i] * ratio)
        # Networks
        self.conv1 = nn.Conv2d(input_dim, initial_dim, kernel_size=7, stride=2, padding=3)
        # N = (W - F + 2P) / S + 1
        # N=（W-7+2*3）/2+1=（W-7+6+2）/2=（W+1）/2
        self.bn1 = norm_layer(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        if args.pretrain == 'resnet34':
            n_block = [3, 4, 6]
        elif args.pretrain == 'resnet18':
            n_block = [2, 2, 2]
        else:
            raise NotImplementedError       
        self.layer1 = self._make_layer(block, block_dims[0], stride=1, norm_layer=norm_layer, num=n_block[0])  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2, norm_layer=norm_layer, num=n_block[1])  # 1/4
        # self.layer3 = self._make_layer(block, block_dims[2], stride=2, norm_layer=norm_layer, num=n_block[2])  # 1/8 注意应该只到四分之一~
        # self.final_conv = conv1x1(block_dims[2], output_dim)#1x1卷积核实现维度的变换
        self.final_conv = conv1x1(block_dims[1], output_dim)#1x1卷积核实现维度的变换(注意输入维度变化)
        self._init_weights(args)

        gwp_debug=0;

    def _init_weights(self, args):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 在torchvision中加载预训练模型（提供ResNet18和ResNet34的预训练模型）
        if self.init_weight:
            from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
            if args.pretrain == 'resnet18':
                pretrained_dict = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            else:
                pretrained_dict = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if self.input_dim == 6:
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        pretrained_dict[k] = torch.cat((v, v), dim=1)
            
            if self.input_dim == 5:#如果输入维度为5，采用boosrand策略来初始化
                for k, v in pretrained_dict.items():
                    if k == 'conv1.weight':
                        # 获取预训练模型中的 conv1 权重
                        pretrained_weight = v  # 形状是 [64, 3, 7, 7]
                        # 按通道求平均，计算每个卷积核的平均值，得到的形状是 [64, 1, 7, 7]
                        avg_weight = pretrained_weight.mean(dim=1, keepdim=True)                        
                        # 将平均值扩展到 5 个通道
                        expanded_weight = avg_weight.repeat(1, 5, 1, 1)  # 形状变成 [64, 5, 7, 7]                        
                        # 将扩展后的权重赋值给 conv1.weight
                        pretrained_dict[k] = expanded_weight
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
        

    def _make_layer(self, block, dim, stride=1, norm_layer=nn.BatchNorm2d, num=2):
        layers = []
        layers.append(block(self.in_planes, dim, stride=stride, norm_layer=norm_layer))
        for i in range(num - 1):
            layers.append(block(dim, dim, stride=1, norm_layer=norm_layer))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        # 注意这里的输入x是一个5维的张量，维度分别是batch_size, num_frames, channel, height, width，需要将其转换为4维张量
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1) #[15, 5, 480, 640]

        x = self.relu(self.bn1(self.conv1(x)))#[15, 64, 240, 320]

        for i in range(len(self.layer1)): #第一层stride为1，因此是不会减小特征图的尺寸的 #[15, 64, 240, 320]
            x = self.layer1[i](x)
        for i in range(len(self.layer2)):#第一层stride为2. [15, 128, 120, 160]
            x = self.layer2[i](x)
        # for i in range(len(self.layer3)):
        #     x = self.layer3[i](x)
        
        # Output
        output = self.final_conv(x)#一个1x1的卷积核，实现维度的变换

        _, c2, h2, w2 = output.shape
        return output.view(b, n, c2, h2, w2)
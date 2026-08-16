from torch import nn
import torch
import timm
import tqdm
import numpy as np
import torch.nn.functional as F


class FSTAModule(nn.Module):
    def __init__(self,  in_channels: int, reduction=2, reconstruct=True):

        super(FSTAModule, self).__init__()
        self.reduction = reduction
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        # convD is used to learn selected bases
        self.convD = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)
        # convV is used to adapt and compress the input feature
        self.convV = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)
        # convA is used to learn attentional queries
        self.convA = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.tanh = nn.Tanh()
        # global_adapt is used to adapt and reconstruct the global enhancement feature
        self.global_adapt = nn.Sequential(
                nn.Conv3d(in_channels // self.reduction, in_channels, kernel_size=1),
                nn.BatchNorm3d(in_channels)
            )

        nn.init.constant_(self.global_adapt[1].weight, 0)
        nn.init.constant_(self.global_adapt[1].bias, 0)

    def forward(self, x: torch.Tensor):

        batch_size, c, t, h, w = x.size()
        n = t * h * w
        assert c == self.in_channels, 'input channel not equal!'
        V = self.convV(x)  # (b, c, t, h, w); adapt and compress the input feature;
        D = self.convD(x)  # (b, k, t, h, w); learn k selected bases;
        A = self.convA(x)  # (b, c, t, h, w); learn attentional queries;

        value = V.view(batch_size, c // self.reduction, n) # compressed input feature
        dct_base = D.view(batch_size, c // self.reduction, n) # bases without normalization
        attention_vectors = A.view(batch_size, c // self.reduction, n) # attentional queries
        # we perform tanh activation and vector normalization successively on the convolution result
        # to obtain the selected bases
        dct_base = self.tanh(dct_base)
        dct_base_l2 = torch.norm(dct_base, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        dct_base = torch.div(dct_base, dct_base_l2)
        # acquire K = c//2 frequency domain features
        spectrum = torch.bmm(value, dct_base.permute(0, 2, 1))  # (b, c, k)
        global_descriptors = spectrum.permute(0, 2, 1) # (b, k, c)

        attention_vectors = F.softmax(attention_vectors, dim=1)  # (b, c, n)
        # calculate the global enhancement feature for each local feature
        global_info = global_descriptors.matmul(attention_vectors)  # (b, k, n)
        global_info = global_info.view(batch_size, c // self.reduction, t, h, w) # (b, k, t, h, w)
        global_info = self.global_adapt(global_info) # (b, c, t, h, w)
        y = global_info + x

        return y, dct_base
    
class Model_FSTANet(torch.nn.Module):
    def __init__(self, frame_length, feature_dim=512, out_dim=512, tdmap_stride=2):
        super(Model_FSTANet, self).__init__()
        self.out_dim = out_dim  # the identity feature dim
        # load the pretrained ResNet18
        self.model = timm.create_model('resnet18', pretrained=True)
        # change the last fc with the shape of 512×512
        self.model.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        # there are 64 frames in each dynamic hand gesture video
        self.frame_length = frame_length
        temporal_diff_length = frame_length - 1
        # tdmap lenth = (temporal_diff_frame_length - temporal_kernel_size) // tdmap_stride + 1
        self.tdmap_frame_length = (temporal_diff_length - 3) // tdmap_stride + 1

        # build TD-Flow module from Conv1 of ResNet18
        conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(tdmap_stride, 2, 2), padding=(0, 3, 3), bias=False)
        # reshape 2D conv parameters to 3D
        pretrained_conv1_params = self.model.conv1.weight.data.unsqueeze(1)
        conv1.weight.data = pretrained_conv1_params
        self.model.conv1 = conv1
        # build FSTA_Module
        self.fsta_module = FSTAModule(in_channels=self.model.layer3[0].conv2.out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # calculate the temporal difference map
    def getTemporalDifferenceMap(self, physical_feature):
        physical_feature = physical_feature.view((-1, self.frame_length) + physical_feature.shape[-3:])
        temporal_diff = physical_feature[:, :self.frame_length - 1, :, :, :] - physical_feature[:, 1:self.frame_length, :, :, :]
        temporal_diff = torch.sum(temporal_diff, 2)
        return temporal_diff

    def forward(self, data, label=None):
        # get 3D temporal difference map
        data_diff = self.getTemporalDifferenceMap(data).unsqueeze(dim=1)
        # get the TD-Flow
        x = self.model.conv1(data_diff)
        x = x.permute(0, 2, 1, 3, 4)
        Bt, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = self.model.bn1(x)
        x = F.relu(x)
        x = self.model.maxpool(x)

        # local behavioral feature extracting
        for i in range(2):
            layer_name = "layer" + str(i + 1)
            layer = getattr(self.model, layer_name)
            x = layer(x)

        # global enhancement feature extracting and feature fusion
        x = self.model.layer3[0](x)
        bn, c, h, w = x.size()
        x = x.view(-1, self.tdmap_frame_length, c, h, w).transpose(1, 2).contiguous()
        x, dct_base = self.fsta_module(x)
        x = x.transpose(1, 2).contiguous().view(bn, c, h, w)
        # global behavioral information summarization
        x = self.model.layer3[1](x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        x = x.view(-1, self.tdmap_frame_length, self.out_dim)

        id_feature = torch.mean(x, dim=1, keepdim=False)
        id_feature = torch.div(id_feature, torch.norm(id_feature, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization for AMSoftmax

        return id_feature, dct_base
    

class fsta(torch.nn.Module):

    def __init__(self,numFrames):
        super().__init__()
        self.numFrames = numFrames
        self.backbone = Model_FSTANet(frame_length=numFrames)
        self.embedDims = 512

    def forward(self,x):
        N,C,T,H,W = x.size()
        x = x.permute(0,2,1,3,4).contiguous().view(-1,C,H,W)
        x, _ = self.backbone(x)
        # x = F.relu(self.dense(x))
        return x
    
    def predict(self,
                dataLoader,
                args):
        
        """
        Function to predict embeddings and outputs

        INPUTS:-
        1) dataLoader: The testSet loader with N samples
        2) args: Parsed arguments

        OUPUTS:-
        1) f_theta: Predicted embeddings of shape (N,d)
        """

        embeddings = []
        
        if(args.multi_gpu == 0):
            device = torch.device(args.device)

        for batch_idx, dataSample in enumerate(tqdm.tqdm(dataLoader,colour='yellow')):

            x = dataSample['data'].to(device)
            y_id = dataSample['label'].to(device)

            self.eval()
            with torch.set_grad_enabled(False):
                f_theta = self.forward(x)
            
            for elemEmbeddings in f_theta.detach().cpu().numpy():
                embeddings.append(elemEmbeddings)                

        return np.array(embeddings)
    

if __name__ == "__main__":

    device = torch.device('cuda:0')
    # model = Model_FSTANet(frame_length=64, feature_dim=512, out_dim=512).to(device)
    # print(model)
    model = fsta(40,143).to(device)
    data = torch.randn(32, 3, 40, 128, 128).to(device)
    # data = data.view(-1, 3, 32, 32)
    op1 = model(data)
    print(op1.size())
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from util import GradientReversal

class Flatten(nn.Module):
    '''
        In the current latest PyTorch version 1.12
        Flatten Layer is not available
    '''
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)

class AttentionDecoupleMetric(nn.Module):
    def __init__(self, cams=3):
        super().__init__()
        self.cams = cams
        self.net = models.googlenet(pretrained=True)

        # output N x 480 x 14 x 14
        self.FNet = nn.Sequential(
            self.net.conv1,
            self.net.maxpool1,
            self.net.conv2,
            self.net.conv3,
            self.net.maxpool2,
            self.net.inception3a,
            self.net.inception3b,
            self.net.maxpool3
        )

        # output N x 832 x 14 x 14
        self.incept4e = nn.Sequential(
            self.net.inception4a,
            self.net.inception4b,
            self.net.inception4c,
            self.net.inception4d,
            self.net.inception4e
        )

        # output N x 1024 x 7 x 7
        self.incept5b = nn.Sequential(
            self.net.maxpool4,
            self.net.inception5a,
            self.net.inception5b
        )

        # output N x 1024 x 1 x 1
        self.GNet = nn.Sequential(
            self.net.inception4a,
            self.net.inception4b,
            self.net.inception4c,
            self.net.inception4d,
            self.net.inception4e,
            self.net.maxpool4,
            self.net.inception5a,
            self.net.inception5b,
            self.net.avgpool
        )

        self.CAMs = nn.ModuleList([nn.Sequential(
                                                    nn.AdaptiveAvgPool2d(1),
                                                    Flatten(),
                                                    nn.Linear(480, 64, bias=False),
                                                    nn.ReLU(),
                                                    nn.Linear(64, 480),
                                                    torch.Sigmoid()
                                                ) 
                                    for i in range(self.cams)])

        self.CA_sub = nn.Sequential(nn.Linear(512//self.cams, 256), nn.ReLu(), nn.Linear(256, 128))

        self.CA_sub_adv = nn.Sequential(GradientReversal(), self.CA_sub)

        self.CA_learners = nn.ModuleList([nn.Linear(1024, 512//self.cams) for i in range(self.cams)])

    def OAM(self, x):
        '''
            x size : (batch_size, depth, H, W)
        '''
        batch_size, depth = x.size(0), x.size(1)
        H, W = x.size(2), x.size(3)
        M = torch.ones(batch_size, H*W) / (H*W)
        
        D = torch.zeros(batch_size, H*W, H*W)
        # fill D with D(i*W+j, i'*W+j') = |x[i, j] - x[i', j']|
        # D is diagonal, so just have to do the math for uppper right corner
        for i in range(H):
            for j in range(W):
                for I in range(H):
                    for J in range(W):
                        if I < i or (I == i and J < j):
                            continue
                        D[:, i*W+j, I*W+J] = (x[:, :, i, j] - x[:, :, I, J]).abs().sum(dim=1)
                        D[:, I*W+J, i*W+j] = D[:, i*W+j, I*W+J]

        # normalize D by out-bound edges (by row)
        D = nn.functional.normalize(D, 1, -1)
        # chain multiplication of D
        for i in range(batch_size):
            D[i] = torch.chain_matmul(*([D[i]]*10))
            M[i] = torch.matmul(D[i], M[i])

        M = M.view(batch_size, H, W)
        
        return M


    def forward(self, x):
        batch_size = x.size(0)

        feat = self.FNet(x)
        # incept4e and incept5b are middle layer of GNet
        # then used to calculate OAM
        incept4e = self.incept4e(feat)
        incept5b = self.incept5b(feat)
        oam = (self.OAM(incept4e) + self.OAM(incept5b)) / 2
        # N x 480 x 14 x 14
        caps = [feat * self.CAMs[i](feat).view(batch_size, 1, 1, -1) for i in range(self.cams)]
        # #cams of N x 480 x 14 x 14
        cams = [self.GNet(caps[i]).view(batch_size, -1) for i in range(self.cams)] # each of size N x 1024
        embeddings = [self.CA_learners[i](cams[i]) for i in range(self.cams)]

        adv_reverse = [self.CA_sub_adv(embeddings[i]) for i in range(self.cams)]
        adv = [self.CA_sub(embeddings[i]) for i in range(self.cams)]

        return cams, embeddings, adv, adv_reverse

if __name__ == '__main__':
    googlenet = models.googlenet(pretrained=False)
    summary(googlenet, (3, 512, 512))
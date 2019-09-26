import torch
from model import AttentionDecoupleMetric

batch_k = 2

model = AttentionDecoupleMetric()
optimizer = torch.optim.Adam(model.parameters())

def loss(embeddings):
    num_data = len(embeddings)
    l = 0
    for i in range(num_data):
        for j in range(num_data):
            if i == j: continue
            l += torch.sqrt((embeddings[i] - embeddings[j])**2)
    return l

def loss_metric(embeddings):
    loss = 0
    for embedding in embeddings:
        batch_size = embedding.size(0)
        for group_index in range(batch_size//batch_k):
            for anchor_index in range(batch_k):
                anchor = embedding[group_index*batch_k+anchor_index, :]
                anchor_length = (anchor * anchor).sum()
                for homo_index in range(batch_k):
                    if homo_index == anchor_index:
                        continue
                    homo = embedding[group_index*batch_k+homo_index, :]
                    homo_length = (homo*homo).sum()
                    distance = (anchor * homo).sum() / (anchor_length * homo_length)
                    loss += torch.log(1+torch.exp(-2*(distance-0.5)))
                for heter_index in range((group_index+1)*batch_k, batch_size):
                    heter = embedding[heter_index, :]
                    heter_length = (heter*heter).sum()
                    distance = (anchor * heter).sum() / (anchor_length * heter_length)
                    loss += torch.log(1+torch.exp(70*(distance-0.5)))
    return loss


def train(dataset, device):
    for i, x in enumerate(dataset):
        x = x.to(device)

        cams, embeddings, embeddings_adv = model(x)
        l_ = loss(embeddings)
        l_adv = loss(embeddings_adv)
        l_metric = loss_metric(cams)
        l = l_ + l_adv + l_metric
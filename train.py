import torch
import argparse
import dataset
from model import AttentionDecoupleMetric

def parse_argument():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--img_folder', type=str, default='./data', help='folder of training image files')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--batch_k', type=int, default=4, help='number of samples for a class of a batch')
    parser.add_argument('--num_batch', type=int, default=5000, help='number of batches per epoch')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
    parser.add_argument('--use-gpu', action='store_true', help='use gpu for training')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs plans to train in total')

    return parser.parse_args()

args = parse_argument()
device = torch.device('cuda:0') if args.use_gpu else torch.device('cpu')
batch_k = 2

model = AttentionDecoupleMetric()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-4)

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

def activation_decay(embeddings, lambda1=0.014):
    '''
        embeddings is a list of all the CA(i, *)
    '''
    loss = 0
    for embedding in embeddings:
        batch_size = embedding.size(0)
        loss += embedding.norm()**2
    return lambda1*loss / (2*batch_size*len*(embeddings))

def loss_ntri(net, lambda2=0.25):
    loss = 0
    for layer in net.CA_learners:
        loss += ca_regularization(layer.weight)
    return lambda2*loss

def ca_regularization(w):
    '''
        w should be (out_dim, in_dim)
    '''
    out_dim = w.size(0)
    w_norm = torch.matmul(w, w.transpose(0, 1)) - torch.eye(out_dim)
    w_norm = w_norm * w_norm
    return (w_norm * torch.eye(out_dim)).sum()

def train(dataset, device):
    for i, x in enumerate(dataset):
        x = x.to(device)

        cams, embeddings, adv, adv_reverse = model(x)
        l_ = loss(adv)
        l_adv = loss(adv_reverse)
        l_metric = loss_metric(cams)
        regularization = loss_ntri(model) + activation_decay(embeddings)
        l = l_ + l_adv + l_metric + regularization

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

if __name__ == '__main__':
    data = dataset.ImageFolderWithName(root=args.img_folder)
    data_loader = torch.utils.data.DataLoader(data, batch_sampler=dataset.CustomSampler(data, batch_size=args.batch_size, batch_k=args.batch_k, len=args.num_batch),
                                                num_workers=args.num_workers)
    for i in range(args.epochs):
        train(data_loader, device)
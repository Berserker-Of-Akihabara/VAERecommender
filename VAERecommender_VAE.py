# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:54:03 2020

@author: caster
"""

import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import VAERecommender_prepare_data as ds
import VAERecommender_metric as metric
import time


h_layer = 600
bottleneck_layer = 300
num_epochs = 200
batchSize = 500
lr = 0.001
anneal_max = 0.2



if __name__ == '__main__':
    if torch.cuda.is_available():
        print('CUDA is available,',torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
class Encoder(torch.nn.Module):
    
    def __init__(self, X_layer):
        super(Encoder, self).__init__()
        #self.sl1 = SparseLinear(X_layer, h_layer)
        self.l1 = torch.nn.Linear(X_layer, h_layer)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.l2_mu = torch.nn.Linear(h_layer, bottleneck_layer)
        self.l2_logvar = torch.nn.Linear(h_layer, bottleneck_layer)
        
    def forward(self, X):
        #print(X.size())
        #X = F.normalize(X.to_dense())
        X = F.normalize(X)
        X = self.dropout(X)
        X = self.l1(X)
        #print(X.size())
        #print(X.size())
        X = F.tanh(X)
        mu = self.l2_mu(X)
        logvar = self.l2_logvar(X)
        X = self.reparameterize(mu, logvar)
        return X, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
class Decoder(torch.nn.Module):
    
    def __init__(self, X_layer):
        super(Decoder, self).__init__()
        self.l1 = torch.nn.Linear(bottleneck_layer, h_layer)
        self.l2 = torch.nn.Linear(h_layer, X_layer)
        
    def forward(self, X):
        X = F.tanh(self.l1(X))
        return self.l2(X)                        #!!!!!!!!!!!!!!!!!!!
    
class CVAE(torch.nn.Module):
    
    def __init__(self, X_layer):
        super(CVAE, self).__init__()
        self.anneal_curr = 0
        self.encoder = Encoder(X_layer)
        self.decoder = Decoder(X_layer)
        
    def forward(self, X):
        X, mu, logvar = self.encoder(X)
        return self.decoder(X), mu, logvar
    
def loss(x, x_gt, mu, logvar, anneal):

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    BCE = -torch.mean(torch.sum(F.log_softmax(x, 1) * x_gt, -1))
    #l = torch.nn.MSELoss()
    #BCE = l(x, x_gt)
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    anneal = min(anneal_max, anneal)
    
    return BCE + anneal*KLD

def getMetrics(pr, gt):
    n100 = metric.NDCG_at_k_batch(pr, gt, 100)
    r100 = metric.Recall_at_k_batch(pr, gt, 100)
    r50 = metric.Recall_at_k_batch(pr, gt, 50)
    r20 = metric.Recall_at_k_batch(pr, gt, 20)
    map20 = metric.map_at_k(pr, gt, 20)
    pers = metric.personalization(pr)
    pers20 = metric.personalization_at_k(pr, 20)
    #n100a = metric.NDCG_at_k_batch_a(pr, gt, 100)
    return (n100, r100, r50, r20, map20, pers, pers20)


def train(model, train_dataset, num_epochs = 200, loss_train_history = None, loss_val_history = None, experiment = True):
    print('TRAINING STARTED')
    global loss
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = loss
    for epoch in range(num_epochs):
        model.train()
        if experiment:
            true_train_dataset, validation_dataset = ds.split_dataset(train_dataset, train_frac = 0.95)
            true_train_dataloader = torch.utils.data.DataLoader(true_train_dataset, batch_size = batchSize, shuffle=True)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = batchSize, shuffle=True)
        else:
            true_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchSize, shuffle=True)
        epoch_train_loss = []
        epoch_val_loss = []
        model.anneal_curr = 0
        model.train()
        model.anneal_curr += (2*anneal_max)/num_epochs
        for labels in true_train_dataloader:
            labels = Variable(labels).cuda()
            labels = labels.to_dense()
            optimizer.zero_grad()
            output, mu, logvar = model(labels)
            #print(output.size())
            #print(labels.to_dense().size())
            loss = criterion(output, labels.to_dense(), mu, logvar, model.anneal_curr)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss)
        if experiment:
            model.eval()
            loss_train_history.append((sum(epoch_train_loss)/len(epoch_train_loss)).cpu().detach().numpy())
            print('epoch:', epoch+1,'train loss:', sum(epoch_train_loss)/len(epoch_train_loss))
            n100 = []
            r100 = []
            r50 = []
            r20 = []
            for labels in validation_dataloader:
                labels = Variable(labels).to(device)
                output, mu, logvar = model(labels)
                loss = criterion(output, labels.to_dense(), mu, logvar, model.anneal_curr)
                epoch_val_loss.append(loss)
                metrics = getMetrics(output,labels)
                n100.extend(metrics[0])
                r100.extend(metrics[1])
                r50.extend(metrics[2])
                r20.extend(metrics[3])
            loss_val_history.append((sum(epoch_val_loss)/len(epoch_val_loss)).cpu().detach().numpy())
            print('epoch:', epoch+1,'val loss:', sum(epoch_val_loss)/len(epoch_val_loss))
            print('n100:',np.mean(metrics[0]),'r100:',np.mean(metrics[1]),'r50:',np.mean(metrics[2]),'r20:',np.mean(metrics[3]))
    print('TRAINING FINISHED')


more = 0
less = 0

def splitBatch(batch, frac):
    global more
    global less
    visible_data = []
    heldout_data = []
    batch = batch.to_dense().numpy()
    for i in range(len(batch)):
        np_user = batch[i]
        count_of_readed = np.count_nonzero(np_user)
        #print(count_of_readed)
        if count_of_readed > 1:
            more += 1
            idx_of_readed = np.argwhere(np_user > 0)
            #print('readed_idx:', idx_of_readed)
            count_to_hide = 1 if count_of_readed * frac < 1 else np.around(count_of_readed * frac)
            #print('count_to_hide:', count_to_hide)
            idx_to_hide = idx_of_readed[np.random.choice(count_of_readed, int(count_to_hide))]
            #print('idx_to_hide:', idx_to_hide)
            heldout = np.zeros_like(np_user)
            heldout[idx_to_hide] = np_user[idx_to_hide]
            visible = np_user
            visible[idx_to_hide] = 0
            heldout = torch.Tensor(heldout)
            visible = torch.Tensor(visible)
            visible_data.append(visible)
            heldout_data.append(heldout)
        else:
            less += 1
    visible_data = torch.stack(visible_data, dim = 0)
    heldout_data = torch.stack(heldout_data, dim = 0)
    print(more, less)
    return visible_data, heldout_data
        
def test(model, dataset):
    model.eval()
    print('TEST')
    total = 0
    correct = 0
    n100 = []
    r100 = []
    r50 = []
    r20 = []
    map20 = []
    pers = []
    pers20 = []
    testDataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle=True)
    for labels in testDataloader:
            labels, heldout = splitBatch(labels, 0.2)
            #labels = labels.to_dense()
            labels = Variable(labels).to(device)
            output, mu, logvar = model(labels)

            output = output.round()
            total += np.prod(list(labels.size()))
            correct += torch.sum(torch.eq(labels, output))

            output[labels > 0] = torch.min(output) - 1
            #print(heldout.size())
            #print(output.size())

            metrics = getMetrics(output, heldout)
            n100.extend(metrics[0])
            r100.extend(metrics[1])
            r50.extend(metrics[2])
            r20.extend(metrics[3])
            map20.append(metrics[4])
            pers.append(metrics[5])
            pers20.append(metrics[6])

    print('n100:',np.mean(metrics[0]),'r100:',np.mean(metrics[1]),'r50:',np.mean(metrics[2]),'r20:',np.mean(metrics[3]),'map20:', np.mean(metrics[4]),'pers:', np.mean(metrics[5]),'pers20:', np.mean(metrics[6]))
    '''
    file = open('hey.txt','w')
    for i in range(len(output[0])):
        file.write('{0}\n'.format(output[0][i]))
    file.close()
    '''
    print(correct.item()/total)
    
def loadModel(model, weigths_path = './CVAE_weights_VAE.pth'):
    checkpoint = torch.load(weigths_path)
    model.load_state_dict(checkpoint, strict = True)

def saveModel(model, weigths_path = './CVAE_weights_VAE.pth'):
    torch.save(model.state_dict(), weigths_path)
    
def predict(model, X):
    model.eval()
    inp = Variable(X[0]).to(device)
    output, _, _ = model(inp)
    return output

'''
def getTopNforUser(model, userId, N, dataset, fullInfo = False):
    model.eval()
    userLibNeuronIndexes, sparseRepr = dataset.getUserLib(userId)
    inp = Variable(sparseRepr).to(device).unsqueeze(0)
    output, _, _ = model(inp)
    print(userLibNeuronIndexes)
    print(output.size())
    output[0,userLibNeuronIndexes] = -1
    top_ratings, top_ind = torch.topk(output, N)
    print(top_ratings, top_ind)
    topIds = dataset.compareIndexesWithNeurons(top_ind)
    if fullInfo:
        res = []
        for vnId in topIds[0]:
            res.append(str(dataset.vndbLib.getVNDataByID(vnId)))
        return res
    else:
        return topIds[0]
'''
    
def getVAERatings(model, res_fin, weight, userId, dataset):
        model.eval()
        res = []
        userLibNeuronIndexes, sparseRepr = dataset.getUserLib(userId)
        inp = Variable(sparseRepr).to(device).unsqueeze(0)
        inp = inp.to_dense()
        output, _, _ = model(inp)
        start = time.time()
        output = output.squeeze(0)
        #output = output.cpu().detach().numpy()
        m1 = torch.min(output)
        output = -m1 + output
        m2 = torch.max(output)
        output = output/m2
        output[userLibNeuronIndexes] = -10
        print('vae norm:',time.time() - start)
        start = time.time()
        output = output.cpu().detach().numpy()
        
        #vnIndexes = dataset.compareIndexesWithNeurons(range(len(output)))
        '''
        for i in range(len(output)):
            if vnIndexes[i] not in removedVNIds:
                res[vnIndexes[i]] = output[i]
        '''
        #res = [ds.ScoredVN(vnIndexes[i], output[i], True) for i in range(len(output))]
        res_fin[dataset.vnIds] += weight * output
        '''
        if relevantVNIds:
            nonrelevantIds = np.array(list(set(dataset.vnIds).difference(set(relevantVNIds))))
            res_fin[nonrelevantIds] = -10
        '''
        '''
        for removedVNId in removedVNIds:
            res[removedVNId] = -10
        '''
        print('vae res:', time.time() - start)
        #return res
    
    

if __name__ == '__main__':
    pass
    '''
    dataset = ds.RatingsDataset()
    X_layer = list(dataset[0][0].size())[0]
    print(X_layer)
    model = CVAE().to(device)
    train_dataset, test_dataset = ds.split_dataset(dataset, train_frac = 0.95)
    testDataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batchSize, shuffle=False)
    loss_train_history = []
    loss_val_history = []
    '''
    
'''
train()
plt.plot(range(num_epochs), loss_train_history)
#plt.plot(range(num_epochs), loss_val_history)
plt.show()
test()
torch.save(model.state_dict(), './CVAE_weights_VAE.pth')
'''

#loadModel(model)
'''
print("_______________")
batch = next(iter(testDataloader))
pr = predict(batch)
m1 = torch.min(pr, -1)[0]
pr = -m1.unsqueeze(-1) + pr
m2 = torch.max(pr,-1)[0]
pr = pr/m2.unsqueeze(-1)
print("_______________")
'''


#print(metric.dcg_score(batch, pr, 100))

#print(getTopNforUser(98417, 20))
'''
for title in getTopNforUser(151834, 20, True):
    print(title+'\n')
'''

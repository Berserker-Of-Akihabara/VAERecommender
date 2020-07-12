import bottleneck as bn
import numpy as np
import sklearn.metrics


def NDCG_at_k_batch_a(X_pred, heldout_batch, k=100):
    
    #print(heldout_batch[1].to_dense().size())
    #print(heldout_batch[1].to_dense().nonzero())
    #print((heldout_batch[1].to_dense() != 0).sum(dim=1))
    X_pred = X_pred.cpu().detach().numpy()
    nnz = (heldout_batch.to_dense() != 0).sum(dim=1)
    heldout_batch = heldout_batch.to_dense().cpu().detach().numpy()
    batch_users = X_pred.shape[0]
    #print(batch_users)
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk] * tp).sum(axis=1)
    #print(nnz.size())
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in nnz])
    return DCG / IDCG
    

    pred = X_pred.cpu().detach().numpy()
    gt = heldout_batch.to_dense().cpu().detach().numpy()
    
    gt_idx_sorted = np.argsort(-pred, axis = 1)
    idx_topk = idx_topk_part[np.arange(len(pred))[:, np.newaxis], gt_idx_sorted]

    gains = 2 ** gt - 1
    tp = 1. / np.log2(np.arange(2, k + 2))
    #print(gt_idx_sorted)
    print(pred[gt_idx_sorted].shape)
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    iDCG = gt[gt_idx_sorted] * tp
    return DCG / IDCG


'''
def NDCG_at_k_batch(y_pred, y_true, k = 10):
    """
    Normalized discounted cumulative gain (NDCG) at rank k
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels)
    
    y_score : array-like, shape = [n_samples]
        Predicted scores
    
    k : int
        Rank

    Returns
    -------
    ndcg : float, 0.0 ~ 1.0
    """
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.to_dense().cpu().detach().numpy()
    actual = dcg_at_k(y_pred, y_true, k)
    best = dcg_at_k(y_true, y_true, k) 
    ndcg = actual / best
    return ndcg
'''
'''
def NDCG_at_k_batch(y_pred, y_true, k = 10):
    """
    Normalized discounted cumulative gain (NDCG) at rank k
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels)
    
    y_score : array-like, shape = [n_samples]
        Predicted scores
    
    k : int
        Rank

    Returns
    -------
    ndcg : float, 0.0 ~ 1.0
    """
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    actual = dcg_at_k(y_pred, y_true, k)
    best = dcg_at_k(y_true, y_true, k) 
    ndcg = actual / best
    return ndcg
'''

def NDCG_at_k_batch(y_pred, y_true, k = 10):
    return [sklearn.metrics.ndcg_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), k)]


def dcg_at_k(y_pred, y_true, k = 10):
    """
    Discounted cumulative gain (DCG) at rank k
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels)
    
    y_score : array-like, shape = [n_samples]
        Predicted scores
    
    k : int
        Rank

    Returns
    -------
    dcg : float
    """
    order = np.argsort(y_pred, axis = 1)[:,::-1]
    y_true_curr = y_true[np.arange(len(order))[:, np.newaxis],
                       order[:, :k + 1]]
    gains = 2 ** y_true_curr - 1
    discounts = 1. / np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains * discounts, axis = 1)
    #print(dcg.shape)
    return dcg
    

def Recall_at_k_batch(y_pred, y_true, k=100):
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    
    batch_users = y_pred.shape[0]
    '''
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    '''
    
    order = np.argsort(y_pred, axis = 1)[:,::-1]
    y_pred_binary = np.zeros_like(y_pred, dtype=bool)
    y_pred_binary[np.arange(len(order))[:, np.newaxis], order[:, :k + 1]] = True
    
    y_true_binary = (y_true > 0)
    tmp = (np.logical_and(y_true_binary, y_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, y_true_binary.sum(axis=1))
    return recall

def ap_at_k(y_pred_idx, y_true, k):
    
    n_hit = 0
    precision = 0
    relevant_idx = np.where(y_true > 0)[0]
    countOfRelevant = relevant_idx.shape[0]
    
    #print('c:',countOfRelevant)
    #print('r:',relevant_idx)
    #print('p:',y_pred_idx)
    for j in range(min(countOfRelevant, k)):
        if y_pred_idx[j] in relevant_idx:
            n_hit += 1.
            precision += n_hit / (j + 1)
    #print('n:',n_hit)
    if n_hit != 0:
        precision /= n_hit
    else:
        precision = 0
    return precision

def map_at_k(y_pred, y_true, k):
    y_pred = y_pred.cpu().detach().numpy()
    y_true = y_true.cpu().detach().numpy()
    res = []
    y_pred_idx = np.argsort(y_pred, axis = 1)[:,::-1][:,:k + 1]
    for i in range(len(y_true)):
        res.append(ap_at_k(y_pred_idx[i], y_true[i], k))
    return np.mean(res)

def personalization(y_pred):
    y_pred = y_pred.cpu().detach().numpy()
    cos_sim = sklearn.metrics.pairwise.cosine_similarity(y_pred)
    for i in range(len(cos_sim) - 1):
        cos_sim[i,:i+1] = 0
    cos_sim[-1,:] = 0
    return 1 - np.sum(cos_sim)/np.sum(np.arange(len(cos_sim)))

def personalization_at_k(y_pred,k):
    print(k)
    y_pred = y_pred.cpu().detach().numpy()
    y_pred_idx_sorted = np.argsort(y_pred, axis = 1)[:,::-1]
    y_pred_curr = y_pred
    y_pred_curr[np.arange(len(y_pred_idx_sorted))[:, np.newaxis], y_pred_idx_sorted[:, k:]] = 0
    cos_sim = sklearn.metrics.pairwise.cosine_similarity(y_pred_curr)
    for i in range(len(cos_sim) - 1):
        cos_sim[i,:i+1] = 0
    cos_sim[-1,:] = 0
    return 1 - np.sum(cos_sim)/np.sum(np.arange(len(cos_sim)))



'''
a = np.array([10, 0, 30, 40, 20])
i = bn.argpartition(a, kth=2)
print(i)
print(a[i])
'''
'''
a = np.asarray([[1,2,5],[3,4,6]])
i = np.asarray([[0,1],[1,0]])
print(a[np.arange(len(i))[:, np.newaxis], i[:,:1]])
'''

if __name__ == '__main__':
    import torch
    
    pred = torch.Tensor([[3, 2 , 3, 0, 1, 2]])
    true = torch.Tensor([[3, 3 , 2, 2, 0 , 1]])
    print(sklearn.metrics.ndcg_score(true, pred, k=6), m.dcg_score(true, pred, k=6))
import torch.nn as nn
import torch
import torch.nn.functional as F

def neg_log_mean_loss(y_true, y_pred):
	eps = 1e-6
	#  yt*log(yp) / |y_t|. missing preds
	pos = - torch.sum(y_true * torch.log(y_pred+eps),1) / torch.maximum(eps, torch.sum(y_true, 1)) 
	# wrong preds
	neg = torch.sum((1-y_true) * (1-y_pred+eps), 1)  / torch.maximum(eps, torch.sum(1-y_true, 1))
	neg = - torch.log(1 - neg + eps)
	loss = pos+ 40*neg
	return loss 

def neg_log_mean_mid_loss(y_true, y_pred):
	pos = - torch.sum(y_true * torch.log(y_pred)) / torch.maximum(1e-6, torch.sum(y_true))
	neg = torch.sum((1-y_true) * y_pred) / torch.maximum(1e-6, torch.sum(1-y_true))
	mid = 0.1
	x = torch.abs(neg - mid)
	neg = - torch.log(1 - x + 1e-6)
	return pos + neg

def cpu_mid_loss(y_true,y_pred,mid=0,pi=0.1,**kwargs):
    eps = 1e-6
    y_true=torch.cast(y_true, 'float32')
    pos = torch.sum(y_true * y_pred, 1) / torch.maximum(eps, torch.sum(y_true, 1))
    pos = - torch.log(pos + eps)
    neg = torch.sum((1-y_true) * y_pred, 1) / torch.maximum(eps, torch.sum(1-y_true, 1))
    neg = torch.abs(neg-mid) 
    neg = - torch.log(1 - neg + eps)
    return torch.mean(pi*pos + neg)

theta = lambda t: (torch.sign(t)+1.)/2.
posmargin = 0.7
negmargin = 0.6

def margin_loss(y_true, y_pred):
	eps = 1e-7
	λpos = 1 - theta(y_true-posmargin)*theta(y_pred - posmargin)
	λneg = 1 - theta(1-negmargin-y_true)*theta(1-negmargin-y_pred)
	pos = - torch.sum( λpos * y_true * torch.log(y_pred+eps),1)
	neg = - torch.sum( λneg * (1-y_true) * torch.log(1-y_pred+eps), 1) /1.5
	return pos+neg
	
# mlmc loss : not good. ~78
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_true = torch.cast(y_true, dtype=tf.float32)
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = torch.concatenate([y_pred_pos, zeros], axis=-1)
    # neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)/1.1
    # FIXME: 函数目前还有些问题
    neg_loss = torch.reduce_logsumexp(y_pred_neg-0.3, axis=-1) # Done mlmc_mid :not good.
    pos_loss = torch.reduce_logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss



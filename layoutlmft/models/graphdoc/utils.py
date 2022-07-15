import torch
import numpy as np
from torch.nn import functional as F
import itertools
from operator import itemgetter

def align_logits(logits):
    batch_size = len(logits)
    max_length = max([_.shape[0] for _ in logits])
    dim = logits[0].shape[1]

    aligned_logits = torch.full((batch_size, max_length, dim), -100, dtype=logits[0].dtype, device=logits[0].device)
    for batch_idx, logits_pb in enumerate(logits):
        aligned_logits[batch_idx, :logits_pb.shape[0]] = logits_pb

    return aligned_logits

def extract_merge_feats_v2(bbox_features, items_polys_idxes, classify_logits):
    l_lst = [sum([len(t) for t in items_polys_idxes_bi]) for items_polys_idxes_bi in items_polys_idxes]
    l_max = max(l_lst)
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    vocab_len = classify_logits.shape[-1]
    entity_features = torch.zeros((B, C + vocab_len, l_max), dtype=dtype, device=device)
    items_polys_idxes_batch = [list(itertools.chain(*items_polys_idxes_bi)) for items_polys_idxes_bi in items_polys_idxes]
    for b_i in range(B):
        entity_index = torch.tensor(items_polys_idxes_batch[b_i], dtype=torch.long, device=device)
        temp_f = bbox_features[b_i, entity_index + 1]  # entity_index + 1: to remove 1st global image
        if len(classify_logits[b_i][1:][entity_index]) > 0:
            classify_class = torch.argmax(classify_logits[b_i][1:][entity_index], dim=-1) # [1:] to remove 1st global image
            classify_encode = F.one_hot(classify_class, num_classes=vocab_len)
            entity_features[b_i, C:, :len(entity_index)] = classify_encode.permute(1, 0)
        entity_features[b_i, :C, :len(entity_index)] = temp_f.permute(1, 0)
        
    merge_mask = torch.zeros((B, l_max), dtype=dtype, device=device)
    for b_i in range(B):
        merge_mask[b_i, :l_lst[b_i]] = 1
    return entity_features, merge_mask

def extract_merge_feats(bbox_features, items_polys_idxes, classify_logits=None):
    l_lst = [sum([len(t) for t in items_polys_idxes_bi]) for items_polys_idxes_bi in items_polys_idxes]
    l_max = max(l_lst)
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    entity_features = torch.zeros((B, C, l_max), dtype=dtype, device=device)
    items_polys_idxes_batch = [list(itertools.chain(*items_polys_idxes_bi)) for items_polys_idxes_bi in items_polys_idxes]
    for b_i in range(B):
        entity_index = torch.tensor(items_polys_idxes_batch[b_i], dtype=torch.long, device=device)
        temp_f = bbox_features[b_i, entity_index + 1]  # entity_index + 1: to remove 1st global image
        entity_features[b_i, :C, :len(entity_index)] = temp_f.permute(1, 0)
        
    merge_mask = torch.zeros((B, l_max), dtype=dtype, device=device)
    for b_i in range(B):
        merge_mask[b_i, :l_lst[b_i]] = 1
    return entity_features, merge_mask


def parse_merge_labels(bbox_features, items_polys_idxes):
    B, C, device, dtype = bbox_features.shape[0], bbox_features.shape[-1], bbox_features.device, bbox_features.dtype
    l_lst = [sum([len(t) for t in items_polys_idxes_bi]) for items_polys_idxes_bi in items_polys_idxes]
    l_max = max(l_lst)
    merge_labels = torch.zeros((B, l_max, l_max), dtype=dtype, device=device) - 1
    for b_i in range(B):
        items_polys_idxes_bi = items_polys_idxes[b_i]
        items_len_lst = [len(t) for t in items_polys_idxes_bi]
        for items_i, items in enumerate(items_polys_idxes_bi):
            items_label = torch.zeros((l_max), dtype=dtype, device=device)
            items_label[sum(items_len_lst[:items_i]):sum(items_len_lst[:items_i + 1])] = 1
            merge_labels[b_i, :, sum(items_len_lst[:items_i]):sum(items_len_lst[:items_i + 1])] = items_label[:, None]
    merge_label_mask = torch.zeros((B, l_max, l_max), dtype=dtype, device=device)
    for b_i, l in enumerate(l_lst):
        merge_label_mask[b_i, :l, :l] = 1
    return merge_labels, merge_label_mask

def select_items_entitys_idx(vocab, classify_logits, attention_mask):
    select_class_idxes = vocab.words_to_ids(["NAME", "CNT", "PRICE", "PRICE&CNT", "CNT&NAME"])
    B = classify_logits.shape[0]
    batch_select_idxes = [[[]] for _ in range(B)]
    for b_i in range(B):
        logit = classify_logits[b_i][attention_mask[b_i].bool()][1:] # remove first whole_image_box, [0, 0, 512, 512]
        pred_class_lst = torch.argmax(logit, dim=1)
        for box_i, pred_class in enumerate(pred_class_lst):
            if pred_class in select_class_idxes:
                batch_select_idxes[b_i][0].append(box_i)
    return batch_select_idxes

def decode_merge_logits(merger_logits, valid_items_polys_idxes, classify_logits, vocab):
    batch_len = [len(t[0]) for t in valid_items_polys_idxes]
    batch_items_idx = []
    for batch_i, logit in enumerate(merger_logits):
        proposal_scores = [[[], []] for _ in range(batch_len[batch_i])] # [idx, idx_score]
        valid_logit = logit[:batch_len[batch_i], :batch_len[batch_i]]
        # select specific classes for merge decode
        yx = torch.nonzero(valid_logit > 0)
        for y, x in yx:
            score_relitive_idx = y
            score_real_idx = valid_items_polys_idxes[batch_i][0][score_relitive_idx]
            proposal_scores[x][0].append(score_real_idx)
            proposal_scores[x][1].append(valid_logit[y, x])
        items = nms(proposal_scores, cal_score='mean')
        batch_items_idx.append(items)
    return batch_items_idx

def nms(proposal_scores, cal_score='mean'):
    proposals = []
    confidences = []
    for p_s in proposal_scores:
        if len(p_s[0]) > 0:
            if cal_score == 'mean':
                score = torch.tensor(p_s[1]).sigmoid().mean()
            else: # multify
                score = torch.tensor(p_s[1]).sigmoid().prod()
            if p_s[0] not in proposals:
                proposals.append(p_s[0])
                confidences.append(score)
            else:
                idx = proposals.index(p_s[0])
                confidences[idx] = max(confidences[idx], score)
    # nms
    unique_proposal_confidence = list(zip(proposals, confidences))
    sorted_proposals_confidence = sorted(unique_proposal_confidence, key=itemgetter(1), reverse=True)
    sorted_proposal = [t[0] for t in sorted_proposals_confidence]
    exist_flag_lst = [True for _ in range(len(sorted_proposal))]
    output_proposals = []
    for pro_i, pro in enumerate(sorted_proposal):
        if exist_flag_lst[pro_i]:
            output_proposals.append(pro)
            for pro_j, tmp_pro in enumerate(sorted_proposal[pro_i + 1:]):
                if overlap(pro, tmp_pro):
                    exist_flag_lst[pro_i + pro_j + 1] = False

    return output_proposals

def overlap(lst1, lst2):
    union_len = len(set(lst1 + lst2))
    if union_len == len(lst1) + len(lst2):
        return False
    else:
        return True


def cal_tp_total(batch_pred_lst, batch_gt_lst, device):
    batch_tp_pred_gt_num = []
    for pred_lst, gt_lst in zip(batch_pred_lst, batch_gt_lst):
        pred_len = len(pred_lst)
        gt_len = len(gt_lst)
        tp = 0
        for pred in pred_lst:
            if pred in gt_lst:
                tp += 1
        batch_tp_pred_gt_num.append([tp, pred_len, gt_len])
    batch_tp_pred_gt_num = torch.tensor(batch_tp_pred_gt_num, device=device)
    return batch_tp_pred_gt_num

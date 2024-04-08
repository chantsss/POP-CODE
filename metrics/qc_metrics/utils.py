from typing import Optional, Tuple

import torchfrom torch_scatter import gather_csrfrom torch_scatter import segment_csr


def topk(
        max_guesses: int,
        pred: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        joint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    max_guesses = min(max_guesses, pred.size(1))
    if max_guesses == pred.size(1):
        if prob is not None:
            prob = prob / prob.sum(dim=-1, keepdim=True)
        else:
            prob = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred, prob
    else:
        if prob is not None:
            if joint:
                if ptr is None:
                    inds_topk = torch.topk((prob / prob.sum(dim=-1, keepdim=True)).mean(dim=0, keepdim=True),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = inds_topk.repeat(pred.size(0), 1)
                else:
                    inds_topk = torch.topk(segment_csr(src=prob / prob.sum(dim=-1, keepdim=True), indptr=ptr,
                                                       reduce='mean'),
                                           k=max_guesses, dim=-1, largest=True, sorted=True)[1]
                    inds_topk = gather_csr(src=inds_topk, indptr=ptr)
            else:
                inds_topk = torch.topk(prob, k=max_guesses, dim=-1, largest=True, sorted=True)[1]
            pred_topk = pred[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob[torch.arange(pred.size(0)).unsqueeze(-1).expand(-1, max_guesses), inds_topk]
            prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)
        else:
            pred_topk = pred[:, :max_guesses]
            prob_topk = pred.new_ones((pred.size(0), max_guesses)) / max_guesses
        return pred_topk, prob_topk


def valid_filter(
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        ptr: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
                                                       torch.Tensor, torch.Tensor]:
    if valid_mask is None:
        valid_mask = target.new_ones(target.size()[:-1], dtype=torch.bool)
    if keep_invalid_final_step:
        filter_mask = valid_mask.any(dim=-1)
    else:
        filter_mask = valid_mask[:, -1]
    pred = pred[filter_mask]
    target = target[filter_mask]
    if prob is not None:
        prob = prob[filter_mask]
    valid_mask = valid_mask[filter_mask]
    if ptr is not None:
        num_nodes_batch = segment_csr(src=filter_mask.long(), indptr=ptr, reduce='sum')
        ptr = num_nodes_batch.new_zeros((num_nodes_batch.size(0) + 1,))
        torch.cumsum(num_nodes_batch, dim=0, out=ptr[1:])
    else:
        ptr = target.new_tensor([0, target.size(0)])
    return pred, target, prob, valid_mask, ptr

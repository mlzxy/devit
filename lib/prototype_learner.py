import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import random
from torch import distributed as dist
import torch.multiprocessing as mp


def sinkhorn(C, iterations=10, epsilon=0.01):
    a = torch.ones(C.shape[0], device=C.device, requires_grad=False) / C.shape[0]
    b = torch.ones(C.shape[1], device=C.device, requires_grad=False) / C.shape[1]

    # Initialise approximation vectors in log domain
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    # Stopping criterion
    # threshold = 1e-1

    def _log_boltzmann_kernel(u, v, C):
        kernel = -C + u.unsqueeze(1) + v.unsqueeze(0)
        kernel /= epsilon  # scaling
        return kernel

    # Sinkhorn iterations
    for i in range(iterations):
        # u0, v0 = u, v

        # u^{l+1} = a / (K v^l)
        K = _log_boltzmann_kernel(u, v, C)
        u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=1)
        u = epsilon * u_ + u

        # v^{l+1} = b / (K^T u^(l+1))
        K_t = _log_boltzmann_kernel(u, v, C).transpose(-2, -1)
        v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=1)
        v = epsilon * v_ + v

        # Size of the change we have performed on u
        # diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(
        #     torch.abs(v - v0), dim=-1
        # )
        # mean_diff = torch.mean(diff)

        # if mean_diff.item() < threshold:
        #     break

    # Transport plan pi = diag(a)*K*diag(b)
    K = _log_boltzmann_kernel(u, v, C)
    pi = torch.exp(K)

    # Sinkhorn distance
    cost = torch.sum(pi * C, dim=(-2, -1))

    return cost, pi


def mask_sinkhorn(
    C,
    mask,
    row_group_size,
    col_group_size,
    iterations=10,
    epsilon=0.01,
    return_cost=False,
):
    mask = mask.bool()
    a = (
        torch.ones(C.shape[0], device=C.device, requires_grad=False)
        / row_group_size.float()
    )
    b = (
        torch.ones(C.shape[1], device=C.device, requires_grad=False)
        / col_group_size.float()
    )

    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    inf_mask = torch.zeros_like(C)
    inf_mask[~mask] = -torch.inf

    def _log_boltzmann_kernel(u, v, C):
        kernel = -C + u.unsqueeze(1) + v.unsqueeze(0)
        kernel /= epsilon  # scaling
        return kernel * mask + inf_mask

    for i in range(iterations):
        K = _log_boltzmann_kernel(u, v, C)
        u_ = torch.log(a + 1e-8) - torch.logsumexp(K, dim=1)
        u_ = torch.nan_to_num(u_, posinf=0)
        u = epsilon * u_ + u

        K_t = _log_boltzmann_kernel(u, v, C).transpose(-2, -1)
        v_ = torch.log(b + 1e-8) - torch.logsumexp(K_t, dim=1)
        v_ = torch.nan_to_num(v_, posinf=0)
        v = epsilon * v_ + v

    K = _log_boltzmann_kernel(u, v, C)
    pi = torch.exp(K)

    if return_cost:
        cost = torch.sum(pi * C, dim=(-2, -1))
        return cost, pi
    else:
        return pi


# Utility


class PrototypeLearner(nn.Module):
    def __init__(
        self,
        num_classes,
        num_prototypes,
        embed_dim,
        queue_size,
        momentum=0.02,
        seed=1,
        iterations=5,
        epsilon=1e-2,
        normalize=True
    ):
        super().__init__()
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.iterations = iterations
        self.epsilon = epsilon
        self.normalize = normalize
        self.max_queue_size = queue_size
        class_ids = list(range(num_classes))
        if self.world_size > 1:
            random.seed(seed)
            random.shuffle(class_ids)
            self.full_class_ids = class_ids
            self.rank2classid = {
                rank: list(lst)
                for rank, lst in enumerate(
                    np.array_split(class_ids, self.world_size)
                )  #! todo: switch to array split
            }
            self.classid2rank = {
                classid: rank
                for rank, lst in self.rank2classid.items()
                for classid in lst
            }
            # print(self.rank2classid)
            self.class_ids = self.rank2classid[self.rank]
            self.max_local_prototype_size = max(
                [len(lst) * num_prototypes for lst in self.rank2classid.values()]
            )
        else:
            self.full_class_ids = class_ids
            self.class_ids = class_ids
            self.classid2rank = {c: 0 for c in class_ids}
            self.rank2classid = {0: class_ids}
            self.max_local_prototype_size = len(class_ids) * num_prototypes
        del class_ids

        
        init_prototypes = torch.zeros(num_classes * num_prototypes, embed_dim)
        nn.init.xavier_uniform_(init_prototypes)

        if self.normalize:
            init_prototypes = F.normalize(init_prototypes, dim=-1)            

        self.register_buffer("prototypes", init_prototypes)

        self.local_prototypes_begin_ind = 0
        for r in range(self.world_size):
            if r < self.rank:
                self.local_prototypes_begin_ind += len(self.rank2classid[r])
        self.local_prototypes_begin_ind *= self.num_prototypes
        self.local_prototypes_end_ind = (
            self.local_prototypes_begin_ind
            + len(self.rank2classid[self.rank]) * self.num_prototypes
        )
        self.full_class_id_inds = torch.as_tensor(np.argsort(self.full_class_ids))
        self.local_prototypes_class_id = (
            torch.as_tensor(self.class_ids)
            .view(-1, 1)
            .repeat(1, num_prototypes)
            .flatten()
        )
        self.local_prototype_group_size = (
            (
                self.local_prototypes_class_id[:, None]
                == self.local_prototypes_class_id[None, :]
            ).sum(1)
        )
        self.local_queue = torch.zeros(queue_size, embed_dim + 1)
        self.local_queue_size = 0
        self._local_prototypes = None

    @property
    def local_num_classes(self):
        return len(self.class_ids)

    def move_private_buffers_if_needed(self, device):
        for buf_name in [
            "local_queue",
            "local_prototype_group_size",
            "local_prototypes_class_id",
            "full_class_id_inds",
        ]:
            tensor = getattr(self, buf_name)
            if tensor.device != device:
                setattr(self, buf_name, tensor.to(device))

    def update(self, tokens, labels, delay=False):
        """
        tokens: [N, embed_dim]
        labels: [N]
        """
        if self.normalize:
            tokens = F.normalize(tokens, dim=-1)

        device = tokens.device
        self.move_private_buffers_if_needed(device)
        class2data = {}

        for c in torch.unique(labels):
            mask = labels == c
            class2data[int(c.item())] = torch.cat(
                [tokens[mask], labels[mask, None]], dim=1
            )

        rank2data = {}
        for r in range(self.world_size):
            _ = [class2data[c] for c in self.rank2classid[r] if c in class2data]
            rank2data[r] = (
                torch.cat(_)
                if len(_) > 0
                else torch.zeros(0, self.embed_dim + 1, device=device)
            )

        if self.world_size > 1:
            send_sizes = []
            for r in range(self.world_size):
                send_sizes.append(len(rank2data.get(r, [])))
            send_sizes = torch.as_tensor(send_sizes, device=device)
            recv_sizes = torch.zeros(self.world_size, device=device, dtype=torch.int64)
            # indice i of recv_sizes means current node will receive recv_sizes[i] amount of data from node-i
            dist.all_to_all_single(recv_sizes, send_sizes)
            recv_data = [
                torch.zeros(int(v.item()), self.embed_dim + 1, device=device)
                for v in recv_sizes
            ]
            send_data = [rank2data[r] for r in range(self.world_size)]
            dist.all_to_all(recv_data, send_data)
            recv_data = torch.cat(recv_data)
        else:
            recv_data = rank2data[0]

        if len(recv_data) > 0:
            self.local_queue = torch.roll(self.local_queue, len(recv_data), 0)
            self.local_queue[: len(recv_data)] = recv_data
            self.local_queue_size = min(
                self.local_queue_size + len(recv_data), self.max_queue_size
            )

            tokens, labels = (
                self.local_queue[: self.local_queue_size, :-1],
                self.local_queue[: self.local_queue_size, -1],
            )
            label_group_size = (labels[:, None] == labels[None, :]).sum(1)
            mask = self.local_prototypes_class_id[:, None] == labels[None, :]
            # C = torch.cdist(self.local_prototypes, tokens)
            C = -(self.local_prototypes @ tokens.T)
            pi = mask_sinkhorn(
                C,
                mask,
                self.local_prototype_group_size,
                label_group_size,
                iterations=self.iterations,
                epsilon=self.epsilon,
            )
            pi_ = pi / pi.sum(0, keepdim=True)
            new_prototypes = pi_ @ tokens
            local_prototypes = (
                1 - self.momentum
            ) * self.local_prototypes + self.momentum * new_prototypes
            if self.normalize:
                local_prototypes = F.normalize(local_prototypes, dim=-1)
        else:
            local_prototypes = self.local_prototypes

        if delay:
            self._local_prototypes = local_prototypes
        else:
            if self.world_size > 1:
                self.sync_prototypes(local_prototypes)
            else:
                self.prototypes[:] = local_prototypes
    
    def flush_delayed_updates(self):
        if self._local_prototypes is not None:
            if self.world_size > 1:
                self.sync_prototypes(self._local_prototypes)
            else:
                self.prototypes[:] = self._local_prototypes
            self._local_prototypes = None
        
    @property
    def local_prototypes(self):
        return self.prototypes[
                self.local_prototypes_begin_ind : self.local_prototypes_end_ind
            ]

    def sync_prototypes(self, local_prototypes):
        prototypes_lst = []
        device = self.prototypes.device
        for r in range(self.world_size):
            prototypes_lst.append(
                torch.zeros(
                    self.max_local_prototype_size, self.embed_dim, device=device
                )
            )
        local_prototype_size = self.local_prototypes_end_ind - self.local_prototypes_begin_ind
        if local_prototype_size < self.max_local_prototype_size:
            local_prototypes_ = torch.zeros(
                self.max_local_prototype_size, self.embed_dim, device=device
            )
            local_prototypes_[: local_prototype_size] = local_prototypes
            local_prototypes = local_prototypes_

        dist.all_gather(prototypes_lst, local_prototypes)
        for r, p in enumerate(prototypes_lst):
            prototypes_lst[r] = p[: len(self.rank2classid[r]) * self.num_prototypes, :]
        self.prototypes[:] = (
            torch.cat(prototypes_lst)
            .view(self.num_classes, self.num_prototypes, self.embed_dim)[
                self.full_class_id_inds
            ]
            .flatten(0, 1)
        )

    @property
    def values(self):
        return self.prototypes
    
    def predict(self, x, return_score=False):
        return self.predict_(self.prototypes, x, return_score=return_score,
                            normalize=self.normalize, num_classes=self.num_classes)
    
    @staticmethod
    def predict_(prototypes, x, return_score=False, normalize=True, 
                num_classes=60):
        num_prototypes = len(prototypes) // num_classes
        if normalize:
            x = F.normalize(x, dim=-1)
        scores = (x @ prototypes.T).reshape(*x.shape[:-1], num_classes, num_prototypes).max(-1).values
        labels = scores.argmax(-1)
        if return_score:
            return labels, scores
        else:
            return labels


def __test_worker_main(
    rank, world_size, gt_prototypes, initial_prototypes, tokens, labels
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()  # all process sync here

    torch.manual_seed(1)

    chunk_size = len(tokens) // world_size
    assert len(tokens) % world_size == 0

    tokens = tokens[rank * chunk_size : (rank + 1) * chunk_size]
    labels = labels[rank * chunk_size : (rank + 1) * chunk_size]

    plearner = PrototypeLearner(
        num_classes=10,
        num_prototypes=3,
        embed_dim=256,
        queue_size=100,
        momentum=0.1,
        epsilon=0.01,
        iterations=10,
    )
    plearner.local_prototypes.data[:] = initial_prototypes.view(10, 3, 256)[
        plearner.class_ids
    ].flatten(0, 1)
    if rank == 0:
        print(f"(rank={rank}) class_ids = ", plearner.class_ids)

    plearner = plearner.to(rank)
    plearner.update(tokens.to(rank), labels.to(rank), delay=True)
    plearner.flush_delayed_updates()

    if rank == 0:
        assert torch.allclose(plearner.prototypes.cpu(), gt_prototypes.cpu(), atol=1e-5)
        print("(Multi Machine) TEST PASSED!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    test plan:
    1. one-step single machine test
    2. one-step multiple machine test
    3. multi-step single machine test (training)
    """
    TEST = "multi_machine"  # "multi_machine"  # 'single_machine'

    def test_single_machine():
        torch.manual_seed(1)
        labels = torch.randint(0, 10, size=(40,))
        labels[labels == 2] = 0  # create no-2 labels
        tokens = torch.rand(40, 256)

        plearner = PrototypeLearner(
            num_classes=10,
            num_prototypes=3,
            embed_dim=256,
            queue_size=100,
            momentum=0.1,
            epsilon=0.01,
            iterations=10,
        )
        initial_prototypes = plearner.local_prototypes.clone()

        prototypes = plearner.local_prototypes.clone().reshape(10, 3, 256)

        for c in torch.unique(labels).long():
            cls_tks = tokens[labels == c]
            _, pi = sinkhorn(
                torch.cdist(prototypes[c], cls_tks), epsilon=0.01, iterations=10
            )
            pi = pi / pi.sum(0, keepdim=True)
            new_protos = pi @ cls_tks
            prototypes[c] = 0.9 * prototypes[c] + 0.1 * new_protos
        prototypes = prototypes.flatten(0, 1)
        prototypes = F.normalize(prototypes, dim=-1)

        plearner.update(tokens, labels)

        assert torch.allclose(plearner.prototypes, prototypes, atol=1e-5)
        print("(Single Machine) TEST PASSED")

        return prototypes, initial_prototypes, tokens, labels

    if TEST == "single_machine":
        test_single_machine()
    elif TEST == "multi_machine":
        world_size = 2
        gt_prototypes, initial_prototypes, tokens, labels = test_single_machine()
        mp.spawn(
            __test_worker_main,
            args=(world_size, gt_prototypes, initial_prototypes, tokens, labels),
            nprocs=world_size,
            join=True,
        )
    else:
        raise ValueError

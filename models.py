import torch
import torch.nn as nn


class TransE(nn.Module):
    def __init__(self, device, num_entity, num_relation, emb_dim, gamma, seed):
        super(TransE, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.device = device
        self.emb_dim = emb_dim
        self.num_entity = num_entity
        self.num_relation = num_relation

        # initialize entity embeddings
        self.entity_emb = self.initialize_emb(self.num_entity, self.emb_dim)

        # initialize relation embeddings
        self.relation_emb = self.initialize_emb(self.num_relation, self.emb_dim)
        self.relation_emb.weight.data.div_(self.relation_emb.weight.data.norm(p=2, dim=1, keepdim=True))
        # create the loss function
        self.loss_fn = nn.MarginRankingLoss(margin=gamma)

    def initialize_emb(self, num_emb, emb_dim):
        emb_weight_range = 6 / torch.sqrt(torch.tensor(emb_dim))
        emb = nn.Embedding(num_embeddings=num_emb, embedding_dim=emb_dim, device=self.device)
        emb.weight.data.uniform_(-emb_weight_range, emb_weight_range)
        return emb

    def forward(self, pos_triplet, neg_triplet):
        self.entity_emb.weight.data.div_(self.entity_emb.weight.data.norm(p=2, dim=1, keepdim=True))
        pos_distance = self.cal_distance(pos_triplet)
        neg_distance = self.cal_distance(neg_triplet)
        return self.loss_fn(pos_distance, neg_distance, torch.tensor([-1], dtype=torch.int64, device=self.device))

    def cal_distance(self, triplet):
        head = triplet[:, 0]
        relation = triplet[:, 1]
        tail = triplet[:, 2]
        return (self.entity_emb(head) + self.relation_emb(relation) - self.entity_emb(tail)).norm(p=2, dim=1)

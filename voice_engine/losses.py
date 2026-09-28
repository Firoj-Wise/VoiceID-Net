# Multi-Tier Metric Alignment & Relational Representation Loss Engine

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTierMetricLoss(nn.Module):
    """
    Multi-Tier Representation Alignment Loss:
    1. Hyperspherical Cosine Alignment (Unit Sphere Angle)
    2. Normalized Euclidean L2 Distance
    3. Relational Knowledge Distance & Angle Manifold preservation
    """
    def __init__(self, w_cos=1.0, w_l2=0.5, w_dist=0.5, w_angle=0.5):
        super().__init__()
        self.w_cos = w_cos
        self.w_l2 = w_l2
        self.w_dist = w_dist
        self.w_angle = w_angle

    def relational_distance(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        t_dist = torch.cdist(t_emb, t_emb, p=2)
        s_dist = torch.cdist(s_emb, s_emb, p=2)
        t_mu = t_dist[t_dist > 0].mean() + 1e-8
        s_mu = s_dist[s_dist > 0].mean() + 1e-8
        return F.smooth_l1_loss(s_dist / s_mu, t_dist / t_mu)

    def relational_angle(self, s_emb: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        td = t_emb.unsqueeze(0) - t_emb.unsqueeze(1)
        sd = s_emb.unsqueeze(0) - s_emb.unsqueeze(1)
        t_norm = F.normalize(td, p=2, dim=2)
        s_norm = F.normalize(sd, p=2, dim=2)
        t_cos = torch.bmm(t_norm, t_norm.transpose(1, 2))
        s_cos = torch.bmm(s_norm, s_norm.transpose(1, 2))
        return F.smooth_l1_loss(s_cos, t_cos)

    def forward(self, student_emb: torch.Tensor, target_emb: torch.Tensor):
        # 1. Cosine similarity loss
        cos_loss = 1.0 - F.cosine_similarity(student_emb, target_emb, dim=-1).mean()
        
        # 2. Normalized L2 loss
        s_unit = F.normalize(student_emb, p=2, dim=-1)
        t_unit = F.normalize(target_emb, p=2, dim=-1)
        l2_loss = F.mse_loss(s_unit, t_unit)
        
        # 3. Relational topology loss
        if student_emb.size(0) > 2:
            dist_loss = self.relational_distance(s_unit, t_unit)
            angle_loss = self.relational_angle(s_unit, t_unit)
        else:
            dist_loss = torch.tensor(0.0, device=student_emb.device)
            angle_loss = torch.tensor(0.0, device=student_emb.device)
            
        total_loss = (
            self.w_cos * cos_loss + 
            self.w_l2 * l2_loss + 
            self.w_dist * dist_loss + 
            self.w_angle * angle_loss
        )
        
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_cos": cos_loss.item(),
            "loss_l2": l2_loss.item(),
            "loss_dist": dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss,
            "loss_angle": angle_loss.item() if isinstance(angle_loss, torch.Tensor) else angle_loss,
        }


class SubCenterArcFace(nn.Module):
    """
    Sub-Center ArcFace Loss for multi-speaker supervised discriminative learning.
    Outputs modified angular margin logits suitable for standard CrossEntropyLoss.
    """
    def __init__(self, in_features, num_classes, sub_centers=2, s=32.0, m=0.25):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.sub_centers = sub_centers
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes * sub_centers, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        cosine = cosine.view(-1, self.num_classes, self.sub_centers)
        cosine, _ = torch.max(cosine, dim=-1)
        
        if labels is None:
            return cosine * self.s
            
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

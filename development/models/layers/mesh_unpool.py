import torch
import torch.nn as nn

# Sparse implementation
class MeshUnpool(nn.Module):
    def __init__(self, unroll_target):
        super(MeshUnpool, self).__init__()
        self.unroll_target = unroll_target

    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group):
        start, end = group.shape
        padding_rows =  self.unroll_target - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols !=0:
            group = group.sparse_resize_((self.unroll_target, self.unroll_target), 2, 0)
        return group

    def pad_occurrences(self, occurrences):
        padding = self.unroll_target - occurrences.shape[0]
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), 1)
            occurrences = padding(occurrences)
        return occurrences

    def pad_mask(self,mask):
        padding = self.unroll_target - mask.shape[0]
        if padding != 0:
            padding = nn.ConstantPad1d((0, padding), False)
            mask = padding(mask)
        return mask

    def get_src_mask(self,mask, unroll_start):
        src_mask = torch.full((1,unroll_start),True)
        src_mask[:,torch.sum(mask == True):] = False
        return src_mask.unsqueeze(0)

    def forward(self, features, meshes):
        batch_size, nf, edges = features.shape

        groups = []
        masks = []
        for mesh in meshes:
            group,mask = mesh.get_groups()
            groups.append(self.pad_groups(group))
            masks.append(mask)

        unroll_mat = torch.stack(groups)

        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1)

        unroll_mat = unroll_mat.to(features.device)

        dst_masks = torch.cat([self.pad_mask(mask) for mask in masks], dim=0).view(batch_size, self.unroll_target)
        src_masks = torch.cat([self.get_src_mask(mask, edges) for mask in masks], dim=0).view(batch_size, edges)

        padded_features = torch.zeros((batch_size, self.unroll_target, nf), device=features.device)
        padded_features[dst_masks] = features.transpose(2,1)[src_masks]

        for mesh in meshes:
            mesh.unroll_gemm()

        res = torch.bmm(unroll_mat.transpose(2,1), padded_features).transpose(2, 1)
        res = res / occurrences

        return res



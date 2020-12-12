import torch
import torch.nn as nn

# Sparse implementation
class MeshUnpool(nn.Module):
    def __init__(self, unroll_target):
        super(MeshUnpool, self).__init__()
        self.unroll_target = unroll_target

    def __call__(self, features, meshes):
        return self.forward(features, meshes)

    def pad_groups(self, group, unroll_start):
        start, end = group.shape
        padding_rows =  unroll_start - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols !=0:
            padding = nn.ConstantPad2d((0, padding_cols, 0, padding_rows), 0)
            group = padding(group)
        return group

    def pad_sparse_groups(self, group):
        start, end = group.shape
        padding_rows =  self.unroll_target - start
        padding_cols = self.unroll_target - end
        if padding_rows != 0 or padding_cols !=0:
            group.sparse_resize_((self.unroll_target, self.unroll_target), 2, 0)
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

    def forward(self, features, meshes):
        batch_size, nf, edges = features.shape
        groups = [self.pad_groups(mesh.get_groups(), edges) for mesh in meshes]
        unroll_mat = torch.cat(groups, dim=0).view(batch_size, edges, -1)

        sparse_groups = [self.pad_sparse_groups(mesh.get_sparse_groups()) for mesh in meshes]
        sparse_unroll_mat = torch.stack(sparse_groups)

        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1)
        dense_occurrences = occurrences.expand(unroll_mat.shape)


        unroll_mat = unroll_mat / dense_occurrences
        unroll_mat = unroll_mat.to(features.device)

        #sparse_unroll_mat = sparse_unroll_mat / sparse_ocurrences
        sparse_unroll_mat = sparse_unroll_mat.to(features.device)

        for mesh in meshes:
            mesh.unroll_gemm()

        res = torch.matmul(features, unroll_mat)

        masks = torch.cat([self.pad_mask(mesh.get_group_mask()) for mesh in meshes],dim=0).view(batch_size,1,self.unroll_target).expand(-1,nf,-1).transpose(2,1)
        padded_features = torch.zeros((batch_size,self.unroll_target,nf),device=features.device)
        test = padded_features[masks]
        padded_features[masks] = features.transpose(2,1).reshape(-1)

        sparseRes = torch.sparse.mm(sparse_unroll_mat.transpose(1,0), padded_features).transpose(1, 0)
        sparseRes = sparseRes / occurrences.expand(sparseRes.shape)

        return res



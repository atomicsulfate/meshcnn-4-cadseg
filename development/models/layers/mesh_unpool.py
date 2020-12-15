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
        padded_groups = [self.pad_groups(mesh.get_groups(), edges) for mesh in meshes]
        unroll_mat = torch.cat(padded_groups, dim=0).view(batch_size, edges, -1)

        sparse_groups = [self.pad_sparse_groups(mesh.get_sparse_groups()) for mesh in meshes]
        sparse_unroll_mat = torch.stack(sparse_groups)

        occurrences = [self.pad_occurrences(mesh.get_occurrences()) for mesh in meshes]
        occurrences = torch.cat(occurrences, dim=0).view(batch_size, 1, -1)
        dense_occurrences = occurrences.expand(unroll_mat.shape)


        unroll_mat = unroll_mat / dense_occurrences
        unroll_mat = unroll_mat.to(features.device)

        sparse_unroll_mat = sparse_unroll_mat.to(features.device)

        masks = [mesh.get_group_mask() for mesh in meshes]

        dst_masks = torch.cat([self.pad_mask(mask) for mask in masks], dim=0).view(batch_size, self.unroll_target)
        src_masks = torch.cat([self.get_src_mask(mask, edges) for mask in masks], dim=0).view(batch_size, edges)

        padded_features = torch.zeros((batch_size, self.unroll_target, nf), device=features.device)
        padded_features[dst_masks] = features.transpose(2,1)[src_masks]

        for mesh in meshes:
            mesh.unroll_gemm()

        res = torch.matmul(features, unroll_mat)


        sparseRes = torch.bmm(sparse_unroll_mat.transpose(2,1), padded_features).transpose(2, 1)
        sparseRes = sparseRes / occurrences.expand(sparseRes.shape)

        assert torch.allclose(sparseRes,res,rtol=1e-03, atol=1e-06), "Error"
        # diffIndices = torch.nonzero(torch.abs(torch.sub(sparseRes, res)) > 0.00001)
        # if (len(diffIndices) > 0):
        #     print(len(diffIndices), "diffs of", res.shape[0]*res.shape[1]*res.shape[2])
            #assert False, "Tensors are not equal"
        return sparseRes



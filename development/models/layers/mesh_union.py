import torch
from torch.nn import ConstantPad2d

# Sparse implementation
class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.groups = torch.eye(n, device=device)

        indicesDim = torch.arange(0, n, device=device).unsqueeze(0)
        indices = torch.cat((indicesDim, indicesDim.clone()))
        values = torch.ones(n, device=device)
        self.sparseGroups = torch.sparse_coo_tensor(indices, values, (n, n), device=device)

        self.device = device

    def union(self, source, target):
        self.groups[target, :] += self.groups[source, :]
        #self.sparseGroups[target].add_(self.sparseGroups[source]) doesn't work

        self.sparseGroups = self.sparseGroups.coalesce()
        sparseRow = self.sparseGroups[source].coalesce()
        rowIdcs = sparseRow.indices()
        allIdcs = torch.cat(((torch.ones(rowIdcs.shape[1], dtype=torch.long, device = self.device) * target).unsqueeze(0), rowIdcs))
        tmpSparseTensor = torch.sparse_coo_tensor(allIdcs,sparseRow.values(),self.sparseGroups.shape)
        self.sparseGroups.add_(tmpSparseTensor)

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        # This is actually never called.
        return self.sparseGroups[edge_key].to_dense()

    def get_occurrences(self):
        sum = torch.sparse.sum(self.sparseGroups, 0).to_dense()
        assert torch.allclose(sum, torch.sum(self.groups, 0)), "Wrong sum"
        return sum

    def get_groups(self, tensor_mask):
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[tensor_mask, :]

    def get_sparse_groups(self):
        self.sparseGroups = self.sparseGroups.coalesce()
        self.sparseGroups.values().clamp_(0, 1)
        return self.sparseGroups.clone()

    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)
        fe = torch.matmul(features.squeeze(-1), self.groups)

        tensor_mask = torch.from_numpy(mask)

        sparseFe = torch.sparse.mm(self.sparseGroups, features.squeeze(-1).transpose(1, 0))[tensor_mask,:].transpose_(1, 0)

        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        fe = fe / occurrences

        sparseOccurrences = torch.sparse.sum(self.sparseGroups,1).to_dense()[tensor_mask]
        sparseFe = sparseFe / sparseOccurrences # could we broadcast instead of expand??

        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
            sparseFe = padding_b(sparseFe)
        assert torch.allclose(sparseFe, fe), "Tensors are not equal"
        # if (not torch.allclose(sparseFe,fe)):
        #     diffIndices = torch.nonzero(torch.sub(sparseFe,fe) > 0.01)
        #     print(len(diffIndices), "diffs of", fe.shape[0]*fe.shape[1])
        #     assert False, "Tensors are not equal"
        return sparseFe

    def prepare_groups(self, features, mask):
        padded_n = features.shape[1]

        tensor_mask = torch.from_numpy(mask)

        self.sparseGroups.values().clamp_(0, 1)

        self.groups = torch.clamp(self.groups[tensor_mask, :], 0, 1).transpose_(1, 0)

        padding_a = features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)
            self.sparseGroups.sparse_resize_((self.sparseGroups.shape[0],padded_n),2,0)

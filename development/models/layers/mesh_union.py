import torch
from torch.nn import ConstantPad2d

# Sparse implementation
class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average

        indicesDim = torch.arange(0, n, device=device).unsqueeze(0)
        indices = torch.cat((indicesDim, indicesDim.clone()))
        values = torch.ones(n, device=device)
        self.groups = torch.sparse_coo_tensor(indices, values, (n, n), device=device)

        self.device = device

    def union(self, source, target):
        #self.groups[target].add_(self.groups[source]) doesn't work

        self.groups = self.groups.coalesce()
        sparseRow = self.groups[source].coalesce()
        rowIdcs = sparseRow.indices()
        allIdcs = torch.cat(((torch.ones(rowIdcs.shape[1], dtype=torch.long, device = self.device) * target).unsqueeze(0), rowIdcs))
        tmpSparseTensor = torch.sparse_coo_tensor(allIdcs, sparseRow.values(), self.groups.shape)
        self.groups.add_(tmpSparseTensor)

    def remove_group(self, index):
        return

    def get_group(self, edge_key):
        # This is actually never called.
        return self.groups[edge_key].to_dense()

    def get_occurrences(self):
        return torch.sparse.sum(self.groups, 0).to_dense()

    def get_groups(self, tensor_mask):
        self.groups = self.groups.coalesce()
        self.groups.values().clamp_(0, 1)
        return (self.groups.clone(),tensor_mask.clone())

    def rebuild_features_average(self, features, mask, target_edges):
        tensor_mask = torch.from_numpy(mask)
        self.prepare_groups(features, tensor_mask)

        fe = torch.sparse.mm(self.groups, features.squeeze(-1).transpose(1, 0))[tensor_mask, :].transpose_(1, 0)
        occurrences = torch.sparse.sum(self.groups, 1).to_dense()[tensor_mask]
        fe = fe / occurrences

        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
        return fe

    def prepare_groups(self, features, tensor_mask):
        self.groups.values().clamp_(0, 1)

        padded_cols = features.shape[1]

        if self.groups.shape[1] < padded_cols:
            self.groups.sparse_resize_((self.groups.shape[0], padded_cols), 2, 0)

import torch
from torch.nn import ConstantPad2d

# Sparse implementation
class MeshUnion:
    def __init__(self, n, device=torch.device('cpu')):
        self.__size = n
        self.rebuild_features = self.rebuild_features_average
        self.groups = torch.eye(n, device=device)

        indicesDim = torch.arange(0, n, device=device).unsqueeze(0)
        indices = torch.cat((indicesDim, torch.clone(indicesDim)))
        values = torch.ones(n, device=device)
        self.sparseGroups = torch.sparse_coo_tensor(indices, values, (n, n), device=device)

        self.device = device

    def union(self, source, target):
        self.groups[target, :] += self.groups[source, :]
        #self.sparseGroups[target].add_(self.sparseGroups[source]) doesn't work

        targetRow = self.sparseGroups[target].clone()
        srcRow = self.sparseGroups[source].clone()

        self.sparseGroups = self.sparseGroups.coalesce()
        sparseRow = self.sparseGroups[source]
        rowIdcs = torch.clone(sparseRow._indices())
        rowValues = torch.clone(sparseRow._values())
        allIdcs = torch.cat(((torch.ones(rowIdcs.shape[1], dtype=torch.long, device = self.device) * target).unsqueeze(0), rowIdcs))
        self.sparseGroups.add_(torch.sparse_coo_tensor(allIdcs,rowValues,self.sparseGroups.shape))

        assert torch.all(torch.eq(self.sparseGroups[target].to_dense(),targetRow.add(srcRow).to_dense())), "Union wrong"

    def remove_group(self, index):
        # self.sparseGroups = self.sparseGroups.coalesce()
        # sparseRow = self.sparseGroups[index]
        # rowIdcs = torch.clone(sparseRow._indices())
        # rowValues = torch.clone(sparseRow._values())
        # allIdcs = torch.cat(((torch.ones(rowIdcs.shape[1], dtype=torch.long, device = self.device) * index).unsqueeze(0), rowIdcs))
        # self.sparseGroups.sub_(torch.sparse_coo_tensor(allIdcs, rowValues, self.sparseGroups.shape))
        #
        # assert torch.count_nonzero(self.sparseGroups[index].to_dense()) == 0, "Not removed"
        return

    def get_group(self, edge_key):
        return self.groups[edge_key, :]

    def get_occurrences(self):
        return torch.sum(self.groups, 0)

    def get_groups(self, tensor_mask):
        self.groups = torch.clamp(self.groups, 0, 1)
        return self.groups[tensor_mask, :]

    def rebuild_features_average(self, features, mask, target_edges):
        self.prepare_groups(features, mask)
        fe = torch.matmul(features.squeeze(-1), self.groups)
        occurrences = torch.sum(self.groups, 0).expand(fe.shape)
        fe = fe / occurrences

        tensor_mask = torch.from_numpy(mask)
        self.sparseGroups = self.sparseGroups.coalesce() # really needed?
        sparseFe = torch.sparse.mm(self.sparseGroups.transpose(1,0), features.squeeze(-1).transpose(1,0))[tensor_mask, :].transpose_(1,0)
        sparseOccurrences = torch.sparse.sum(self.sparseGroups,0).to_dense()[tensor_mask];
        sparseOccurrences = sparseOccurrences.expand(fe.shape[0],sparseOccurrences.shape[0]) # could we broadcast instead of expand??
        sparseFe = sparseFe / sparseOccurrences
        #sparseFe = sparseFe[:, torch.from_numpy(mask)]

        padding_b = target_edges - fe.shape[1]
        if padding_b > 0:
            padding_b = ConstantPad2d((0, padding_b, 0, 0), 0)
            fe = padding_b(fe)
            sparseFe = padding_b(sparseFe)
        if (not torch.allclose(sparseFe,fe)):
            diffIndices = torch.nonzero(torch.sub(sparseFe,fe) > 0.01)
            print(len(diffIndices), "diffs of", fe.shape[0]*fe.shape[1])
            assert False, "Tensors are not equal"
        return fe

    def prepare_groups(self, features, mask):
        #n = len(self.groupIndices)
        padded_n = features.shape[1]
        # indices = torch.cat((torch.arange(0,n,device=self.device).unsqueeze(0),self.groupIndices.unsqueeze(0)))
        # values = torch.ones(n,device=self.device)
        # self.sparseGroups = torch.sparse_coo_tensor(indices,values,(n,padded_n),device=self.device)

        assert torch.max(self.groups) == 1.0, "Values greater than 1"
        assert torch.min(self.groups) == 0.0, "Values less than 0"

        tensor_mask = torch.from_numpy(mask)

        assert torch.allclose(self.groups, self.sparseGroups.to_dense()), "Error"

        assert torch.allclose(self.groups[tensor_mask,:],self.sparseGroups.to_dense()[tensor_mask,:]), "Error"
        self.groups = torch.clamp(self.groups[tensor_mask, :], 0, 1).transpose_(1, 0)



        padding_a = features.shape[1] - self.groups.shape[0]
        if padding_a > 0:
            padding_a = ConstantPad2d((0, 0, 0, padding_a), 0)
            self.groups = padding_a(self.groups)
            self.sparseGroups.sparse_resize_((padded_n,self.sparseGroups.shape[1]),2,0)

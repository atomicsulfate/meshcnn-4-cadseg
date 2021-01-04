import torch
from meshcnn.models.layers import mesh_conv

class MeshConv(mesh_conv.MeshConv):

    def create_GeMM(self, x, Gi):
        Gishape = Gi.shape
        # pad the first row of  every sample in batch with zeros
        padding = torch.zeros((x.shape[0], x.shape[1], 1), requires_grad=True, device=x.device)
        # padding = padding.to(x.device)
        x = torch.cat((padding, x), dim=2)
        Gi = Gi + 1  # shift

        # first flatten indices
        Gi = self.flatten_gemm_inds(Gi)
        Gi = Gi.view(-1).long()
        #
        odim = x.shape
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(odim[0] * odim[2], odim[1])

        x = torch.index_select(x, dim=0, index=Gi)
        del Gi
        x = x.view(Gishape[0], Gishape[1], Gishape[2], -1)
        x = x.permute(0, 3, 1, 2)

        # apply the symmetric functions for an equivariant conv
        x[:, :, :, 1] += x[:, :, :, 3]
        x[:, :, :, 2] += x[:, :, :, 4]
        x[:, :, :, 3] = torch.abs(x[:, :, :, 1] - 2 * x[:, :, :, 3])
        x[:, :, :, 4] = torch.abs(x[:, :, :, 2] - 2 * x[:, :, :, 4])
        return x
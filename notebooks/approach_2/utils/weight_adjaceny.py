import torch

def weight_adjacency_matrix(coords, faces):
        """
    
        coords : vertex position (nx3)
        faces  : traingulation (nx3) vetex indices for each triangle

        Returns 
                edge_indx
                edge_attr
        """
        faces = faces.type(torch.LongTensor).to(device=faces.device)
        vertice_ax0_1 = (coords[faces[:,0].t() , :].t() - coords[faces[:,1].t() ,:].t()).pow(2).sum(0).t()
        vertice_ax0_2 = (coords[faces[:,0].t() , :].t() - coords[faces[:,2].t() ,:].t()).pow(2).sum(0).t()
        vertice_ax1_0 = (coords[faces[:,1].t() , :].t() - coords[faces[:,0].t() ,:].t()).pow(2).sum(0).t()
        vertice_ax1_2 = (coords[faces[:,1].t() , :].t() - coords[faces[:,2].t() ,:].t()).pow(2).sum(0).t()
        vertice_ax2_0 = (coords[faces[:,2].t() , :].t() - coords[faces[:,0].t() ,:].t()).pow(2).sum(0).t()
        vertice_ax2_1 = (coords[faces[:,2].t() , :].t() - coords[faces[:,1].t() ,:].t()).pow(2).sum(0).t()

        weights = torch.hstack((vertice_ax0_1,vertice_ax0_2,vertice_ax1_0,vertice_ax1_2,vertice_ax2_0,vertice_ax2_1)).sqrt()        
        weights = 1/weights #inverse

        rows = torch.hstack((faces[:, 0].t(), faces[:, 0].t(), faces[:, 1].t(), faces[:, 1].t(), faces[:, 2].t(), faces[:, 2].t()))
        cols = torch.hstack((faces[:, 1].t(), faces[:, 2].t(), faces[:, 0].t(), faces[:, 2].t(), faces[:, 0].t(), faces[:, 1].t()))
        temp_array = torch.cat((rows.unsqueeze(1), cols.unsqueeze(1)), dim=1)
        unique, inverse = torch.unique(temp_array, sorted=True, return_inverse=True, dim=0)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        index = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        weights = weights[index]

        return(unique.t(), weights)
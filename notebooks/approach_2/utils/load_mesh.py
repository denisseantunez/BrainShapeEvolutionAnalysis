import os
import torch
from nilearn import surface

class LoadMesh:
    def load_mesh(self, main_path, id, hemi, device):
        """
        Loads a surface mesh (.surf.gii) using Nilearn.
        Assumes the path points to a GIFTI file with vertex and face data.

        Parameters:
            main_path : str : path to dataset directory
            id        : str : filename of the mesh (e.g. 'lh.white.surf.gii')
            hemi      : str : ignored here, use if naming depends on hemisphere
            device    : str : 'cpu' or 'cuda'

        Returns:
            self.coords : torch.Tensor : vertices
            self.faces  : torch.Tensor : triangle faces
        """
        self.device = device

        path = os.path.join(main_path, id)
        try:
            mesh = surface.load_surf_mesh(path)
            self.coords = torch.from_numpy(mesh.coordinates).to(device=device)
            self.faces = torch.from_numpy(mesh.faces.astype('float')).to(device=device)
        except FileNotFoundError:
            print(f"Mesh file not found at: {path}")
            self.coords = None
            self.faces = None

        # Set dummy attributes to keep compatibility with downstream code
        self.depth = torch.zeros(self.coords.shape[0]).to(device=device)
        self.thickness = torch.zeros(self.coords.shape[0]).to(device=device)
        self.curv = torch.zeros(self.coords.shape[0]).to(device=device)
        self.P = torch.zeros(self.coords.shape[0]).to(device=device)

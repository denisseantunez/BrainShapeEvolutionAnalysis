import torch
import numpy as np
import os
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
import trimesh 

folder_path_out = 'data/reconstructed/'


def store_coefficients(c : np.array, store_dir : str):
    root, file = os.path.split(store_dir)
    os.makedirs(root, exist_ok=True)
    np.save(store_dir, c)

def main() -> None:

    mesh_data_path = 'data/after_alignment/'

    template_spec_data = torch.load('data/after_alignment/spectral_data/sub-001_species-Cercopithecus+cephus_hemi-L.surf.gii_lh.pt')
    U_ref = template_spec_data["eig_vec"]
    X_template = template_spec_data["ali_spe"]
    
    template_mesh_data = torch.load('data/after_alignment/mesh_data/sub-001_species-Cercopithecus+cephus_hemi-L.surf.gii_lh.pt')
    template_coords = template_mesh_data["coords"]
    template_faces = template_mesh_data["faces"]

    spectral_path = os.path.join(mesh_data_path, 'spectral_data')
    mesh_path = os.path.join(mesh_data_path, 'mesh_data')
    target_id = 1
    
    for samp_file in sorted(os.listdir(spectral_path)):
        samp_spec_data = torch.load(os.path.join(spectral_path, samp_file))
        samp_mesh_data = torch.load(os.path.join(mesh_path, samp_file))
        

        X_target = samp_spec_data["ali_spe"]
        v = samp_mesh_data["coords"]
        

        target_tree = cKDTree(v)
        _, indices = target_tree.query(template_coords)

        Z_1 = X_template[:, :3]
        Z_2 = X_target[:, :3]

        print("Z_1.shape[0]:", Z_1.shape[0])
        print("corr_12 shape:", indices.shape)
        print("min/max values in corr_12:", np.min(indices), np.max(indices))
    
        v_proj = coo_matrix((np.ones(Z_1.shape[0]),
                               (np.arange(Z_1.shape[0]), indices)),
                              shape=(Z_1.shape[0], Z_2.shape[0])) @ v
        

        #if X.shape[0] != U_ref.shape[0]:
        #    print(f"Skipping {samp_file} due to shape mismatch: {X.shape} vs {U_ref.shape}")
        #    continue

        print(f"/n{samp_file} shapes: {v.shape} vs {U_ref.T.shape}/n")

        C = U_ref.T @ v_proj

        print(f"/nCoefficients calculated!: {C}, {C.shape}/n")


        filename = os.path.basename(samp_file).replace(".surf.gii_lh.pt", "")

        store_coefficients(C.detach().cpu().numpy(), f'data/aligned_coefficients/{filename}.npy')

        pos_aligned = U_ref @ C.to(dtype=U_ref.dtype)
        mesh_tri_aligned = trimesh.Trimesh(vertices=pos_aligned, faces=template_faces)

        out_dir = os.path.join(folder_path_out, str(target_id))
        os.makedirs(out_dir, exist_ok=True)
        mesh_tri_aligned.export(os.path.join(out_dir, 'reconstructed_aligned.stl'))
        target_id += 1



if __name__ == '__main__':
    main()

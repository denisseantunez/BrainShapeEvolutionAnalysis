#!/usr/bin/env python

"""
Script to perform spectral aligmnet between given and reference brain surface

If this code is useful to you, please cite:

Herve paper
Karthik paper
"""

import os
import argparse
import timeit
import torch
from torch_geometric.data import Data
from utils.load_mesh import LoadMesh
from utils.embedding import Embedding


# parse the input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--ref', required=True, help='directory for the reference brain')
parser.add_argument('-s', '--sub', required=True, help='directory for the subj/to be aligned brain')
parser.add_argument('-o', '--out', required=True, help='outut directory for saving data')
parser.add_argument('--hemi', default='lh', help='hemisphere to align (`lr` or `rh`)')
parser.add_argument('--eig', default=100, help='number of eigenvectors to decompose')
parser.add_argument('--sul', default='True', action='store_true', help='use sulcal depth for alignment matching')
parser.add_argument('--two_step', default='False', action='store_true', help='use first 3 less ambiguous eigenvectors to align')
#parser.add_argument('--gpu', default='False', action='store_true', help='GPU or CPU')
parser.add_argument('--gpu', action='store_true', help='GPU or CPU')

parser.add_argument('--robust', default='False', action='store_true', help='robust vs faster alignment. fast - uses few eigen and partial matching')
parser.add_argument('--verbose', default='False', action='store_true', help='Verbose mode')
args = parser.parse_args()

# set robust vs fast parameters
if args.robust == True:
    matching_samples = [] #  using all points for matching
    matching_mode = 'complete' #partial or complete
    print('Using robust alignment with all points')
else:
    matching_samples = 10000  #  number of points used to find transformation #5000
    matching_mode = 'partial' #partial or complete    
    print('Using faster alignment with {}'.format(matching_samples))

if args.gpu:
    print('Using GPU')
    device = 'cuda'
else:
    print('Using CPU')
    device = 'cpu'

# set the reference, subject and output directories
ref_path, ref = os.path.split(args.ref)
sub_path, sub = os.path.split(args.sub)

out_path = args.out
if not os.path.exists(out_path):
    print('Creating output directory')
    os.makedirs(out_path)
if not os.path.exists(os.path.join(out_path, 'spectral_data')):
    os.makedirs(os.path.join(out_path, 'spectral_data'))
if not os.path.exists(os.path.join(out_path, 'mesh_data')):
    os.makedirs(os.path.join(out_path, 'mesh_data'))

start = timeit.default_timer()
st = start

# check for self alignment
if ref == sub:
    print('Self alignment - Skipping computation')
    ref_data = LoadMesh()
    
    print('Loading {} as reference mesh'.format(ref))
    ref_data.load_mesh(ref_path, ref, args.hemi, device)
    ref_spectral_embedding = Embedding(ref_data)
    print('Computing spectral embedding of {} as reference'.format(ref))
    ref_spectral_embedding.spectral(args.eig)

    spec_data = Data(eig_vec = ref_spectral_embedding.eig_vecs,
            eig_val = ref_spectral_embedding.eig_vals,
            ali_spe = ref_spectral_embedding.X,
            uni_spe = ref_spectral_embedding.X)

    mesh_data = Data(depth = ref_spectral_embedding.depth,
                curv = ref_spectral_embedding.curv,
                thick = ref_spectral_embedding.thickness,
                parc = ref_spectral_embedding.P,
                edge_index = ref_spectral_embedding.edge_index,
                edge_attr = ref_spectral_embedding.edge_attr,
                coords = ref_spectral_embedding.coords,
                #aligned_coords = ref_spectral_embedding.aligned_coords,
                faces = ref_spectral_embedding.faces)
    
    # Save the spectral and mesh data
    spec_data.to('cpu'); mesh_data.to('cpu')
    # check if the value is a tensor and its dtype is float64 and change to float32
    for key, value in spec_data.items():
        if torch.is_tensor(value) and value.dtype == torch.float64:
            spec_data[key] = value.float()
    for key, value in mesh_data.items():
        if torch.is_tensor(value) and value.dtype == torch.float64:
            mesh_data[key] = value.float()
    torch.save(spec_data, os.path.join(out_path, 'spectral_data', ref + '_' + args.hemi + '.pt'))
    torch.save(mesh_data, os.path.join(out_path, 'mesh_data', ref + '_' + args.hemi + '.pt'))
    
else:
    # Load reference mesh and  compute the spectral embedding
    ref_data = LoadMesh()
    print('Loading {} as reference mesh'.format(ref))
    ref_data.load_mesh(ref_path, ref, args.hemi, device)
    ref_spectral_embedding = Embedding(ref_data)
    print('Computing spectral embedding of {} as reference'.format(ref))
    ref_spectral_embedding.spectral(args.eig)

    # Load subject mesh and compute the spectral embedding
    sub_data = LoadMesh()
    print('Loading {} as subject mesh'.format(sub))
    sub_data.load_mesh(sub_path, sub, args.hemi, device)
    sub_spectral_embedding = Embedding(sub_data)
    print('Computing subject spectral embedding of {} as subject'.format(sub))
    sub_spectral_embedding.spectral(args.eig)
    
    # spectral embedding and matching for other scans
    print('Aligning subject {} spectral embedding to {} reference'.format(sub, ref))
    sub_spectral_embedding.align(ref_spectral_embedding, args.eig, matching_samples, args.sul, two_step=args.two_step, matching_mode=matching_mode, verbose = args.verbose)   

    # Save the spectral and mesh data in pytorch format
    spec_data = Data(eig_vec = sub_spectral_embedding.eig_vecs,
                eig_val = sub_spectral_embedding.eig_vals,
                ali_spe = sub_spectral_embedding.X,
                uni_spe = torch.matmul(sub_spectral_embedding.eig_vecs, torch.diag(sub_spectral_embedding.eig_vals ** (-0.5))))

    mesh_data = Data(depth = sub_spectral_embedding.depth,
                curv = sub_spectral_embedding.curv,
                thick = sub_spectral_embedding.thickness,
                parc = sub_spectral_embedding.P,
                edge_index = sub_spectral_embedding.edge_index,
                edge_attr = sub_spectral_embedding.edge_attr,
                coords = sub_spectral_embedding.coords,
                faces = sub_spectral_embedding.faces)

    # Save the spectral and mesh data
    device = 'cpu'
    spec_data.to(device); mesh_data.to(device)
    # check if the value is a tensor and its dtype is float64 and change to float32
    for key, value in spec_data.items():
        if torch.is_tensor(value) and value.dtype == torch.float64:
            spec_data[key] = value.float()
    for key, value in mesh_data.items():
        if torch.is_tensor(value) and value.dtype == torch.float64:
            mesh_data[key] = value.float()

    torch.save(spec_data, os.path.join(out_path, 'spectral_data', sub + '_' + args.hemi + '.pt'))
    torch.save(mesh_data, os.path.join(out_path, 'mesh_data', sub + '_' + args.hemi + '.pt'))

stop = timeit.default_timer()
print('Time taken: ',(stop-start),' s')

print("########################################################")





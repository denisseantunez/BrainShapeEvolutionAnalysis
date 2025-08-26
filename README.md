# Brain Shape Evolution Analysis
Repository with the tools and resources of the project: Analysis of Brain Shape Evolution

<p align="center">
   <img width="580" height="564" alt="RadialTreeThumbnails" src="https://github.com/user-attachments/assets/b03f5b77-1927-4ce9-a446-49cddc25c183" />
</p>


## Objective
Apply **spectral shape analysis** to:
- Describe the shape of **90 brains** from different species.  
- Align all brains in a **common reference frame**.  
- Cluster results and build a **dendrogram of the evolutionary tree**.

## Usage Notes
1. **Unzip** the surface file containing all 90 surfaces.  
2. **Specify paths** in `src/generate_data.py`.  
3. Run:
   ```bash
   python generate_data.py

## Core requirements
- numpy
- scipy
- matplotlib
- pyvista
- trimesh
- tables
- nilearn
- lapy





## Bibliography
[1] Schwartz, E., Nenning, KH., Heuer, K. et al. Evolution of cortical geometry and its link to function, behaviour and ecology. Nat Commun 14, 2252 (2023). https://doi.org/10.1038/s41467-023-37574-x https://github.com/cirmuw/EvolutionOfCorticalShape   
[2] Kubík, T., Guibault, F., Španěl, M., & Lombaert, H. (2025, June 3). ToothForge: Automatic dental shape generation using synchronized spectral embeddings. arXiv.org. https://doi.org/10.48550/arXiv.2506.02702 https://github.com/tiborkubik/toothForge   
[3] Gopinath, K., Desrosiers, C., & Lombaert, H. (2019). Graph Convolutions on Spectral Embeddings for Cortical Surface Parcellation. Medical image analysis, 54, 297–305. https://doi.org/10.1016/j.media.2019.03.012 https://github.com/kharitz/brain-surface-spectral-alignment   
[4] Lombaert, H., Grady, L., Polimeni, J. R., & Cheriet, F. (2013). FOCUSR: feature oriented correspondence using spectral regularization--a method for precise surface matching. IEEE transactions on pattern analysis and machine intelligence, 35(9), 2143–2160. https://doi.org/10.1109/TPAMI.2012.276   
[5] Lombaert, H., Arcaro, M., & Ayache, N. (2015). Brain Transfer: Spectral Analysis of Cortical Surfaces and Functional Maps. Information processing in medical imaging : proceedings of the ... conference, 24, 474–487. https://doi.org/10.1007/978-3-319-19992-4_37   
[6] Reuter, M., Biasotti, S., Giorgi, D., Patanè, G., Spagnuolo, M.: Discrete laplace–beltrami operators for shape analysis and segmentation. Computers & Graphics (2009)

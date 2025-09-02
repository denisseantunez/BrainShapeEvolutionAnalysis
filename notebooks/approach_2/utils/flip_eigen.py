import torch


def flip_eigen_sign(M1, M2, ne, verbose):
        """
        Flip eigen vector signs of M2 such that it matches M1

        M1 = Reference embedding
        M2 = Embedding to flip sign
        ne = number of eigen vectors

        """
        for i in range(0, ne):
            
            #center the mean (zero mean)
            X1 = M1.coords[:,0:3] - M1.coords[:,0:3].mean(0)
            X2 = M2.coords[:,0:3] - M2.coords[:,0:3].mean(0)

            
            #reference
            #weighted barycenters of poles
            w1 = torch.sign(M1.X[:,i]) * (abs(M1.X[:,i]**3))
            
            #positive pole barycenter
            w = w1.clone()
            w[w<0] = 0
            w = w/w.sum(0)
            avgX1p = (X1 * w.unsqueeze(-1)).sum(0)
            del(w)

            #negative pole barycenter
            w = w1.clone()
            w[w>0] = 0
            w = w/w.sum(0)
            avgX1m = (X1 * w.unsqueeze(-1)).sum(0)
            del(w)

            #to transform
            w2 = torch.sign(M2.X[:,i]) * (abs(M2.X[:,i]**3))

            #positive pole barycenter
            w = w2.clone()
            w[w<0] = 0
            w = w/w.sum(0)
            avgX2p = (X2 * w.unsqueeze(-1)).sum(0)
            del(w)

            #negative pole barycenter
            w=w2.clone()
            w[w>0] = 0
            w = w/w.sum(0)
            avgX2m = (X2 * w.unsqueeze(-1)).sum(0)
            
            #distance betweem matched poles
            distp = (avgX1p - avgX2p).pow(2).sum() + (avgX1m - avgX2m).pow(2).sum()
            distm = (avgX1p - avgX2m).pow(2).sum() + (avgX1m - avgX2m).pow(2).sum()

            if distm < distp:
                if verbose:
                    print('Flip ',str(i))
                M2.eig_vecs[:,i] = - M2.eig_vecs[:,i]
                M2.X[:,i] = - M2.X[:,i]
            
        return M2
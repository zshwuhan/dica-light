function [Dest, cest, A, lams, W, vecs] = spectral(SX, K, momtype, c0)
%SPECTRAL Spectral algorithm for topic modeling
%
% [A, lams, W] = spectral(SX, K, momtype, c0)
%
% Input:
%   SX      : sparse M-by-N matrix of word counts X with docs in columns
%   K       : number of topics
%   momtype : either 'lda' for LDA-moments or 'dica' for DICA-cumulants
%   c0      : parameter (only for LDA-moments, i.e. momtype = 'lda')
%
% Output:
%   A      : K-by-M diagonalizing matrix
%   lams   : K-vector with the eigenvalues
%   W      : K-by-M whitening matrix
%   vecs   : cell with the projection vector
%
% Comment: See "estimate_D.m" for estimation of the topic matrix.

% Copyright 2015, Anastasia Podosinnikova


  if ~( strcmp(momtype,'dica') || strcmp(momtype,'lda') )
    error('Wrong momtype')
  end
  
  % compute whitening
  if strcmp(momtype,'dica')
    [W, M1] = compute_S_and_W_dica(SX, K);
  end
  if strcmp(momtype,'lda')
    [W, M1, M2] = compute_S_and_W_lda(SX, K, c0);
  end
  
  u = rand(K,1); u = u/norm(u);
  vecs{1} = W'*u;
  
  if strcmp(momtype,'dica')
    WTW = compute_multiple_wtw_dica(SX, W, K, vecs, M1);
  end
  if strcmp(momtype,'lda')
    WTW = compute_multiple_wtw_lda(SX, W, K, vecs, M1, M2, c0);
  end
  
  [evecs,evals] = eig(WTW);
  thetas = evecs;
  lams = diag(evals);
  A = thetas'*W;
  
  
  
  % Estimation of the parameters (D, c)
  
  M = size(A,2);
  % problem: the pseudo inverse can introduce negative values
  Dest = pinv(A);
  % each column of Dest is estimated up to multiplication by scalar
  % => checking wheter columns have correct signs
  [Dest, signs] = flip_column_signs(Dest);
  % truncate all negative values
  Dest = max(0,Dest);
  
  % before normalizing Dest, compute c
  c = zeros(K,1);
  v = vecs{1};
  for k = 1:K
    if strcmp(momtype,'dica')
      c(k) = 4*(Dest(:,k)'*v*signs(k))^2 / lams(k)^2;
    end
    if strcmp(momtype,'lda')
      c(k) = 4*(Dest(:,k)'*v*signs(k))^2 / lams(k)^2;
    end
  end
  if strcmp(momtype,'lda')
    c = c * (c0*(c0+1)/(c0+2)^2);
  end
  cest = c;
  
  
  % normalize each column of Dest to be in the simplex
  Dest = Dest./repmat(sum(Dest),M,1);
  
end

function [Dest, A, lams, W] = tpm(SX, K, momtype, c0)
%TPM Tensor power method for topic modeling
%
% [A, lams, W] = tpm(SX, K, momtype, c0)
%
% Input:
%   SX      : sparse M-by-N matrix of word counts X with docs in columns
%   K       : number of topics
%   momtype : either 'lda' for LDA-moments or 'dica' for DICA-cumulants
%   c0      : parameter (only for LDA-moments, i.e. momtype = 'lda')
%
% Output:
%   Dest : estimation of the topic matrix
%   A    : K-by-M diagonalizing matrix
%   lams : K-vector with robust eigenvalues
%   W    : K-by-M whitening matrix
%
% Comment: See "estimate_D.m" for estimation of the topic matrix.

% Copyright 2015, Anastasia Podosinnikova

  if ~( strcmp(momtype,'lda') || strcmp(momtype,'dica') )
    error('Wrong momtype')
  end

  % whitening and precomputation
  if strcmp(momtype,'dica')
    [W, M1] = compute_S_and_W_dica(SX,K);
    precomputed = precompute_for_wtw(SX, W, momtype, M1);
  end
  if strcmp(momtype,'lda')
    [W, M1, M2] = compute_S_and_W_lda(SX,K,c0);
    precomputed = precompute_for_wtw(SX, W, momtype, M1, M2);
  end
  
  
  thetas = zeros(K,K); % robust eigenvectors
  lams   = zeros(K,1); % robust eigenvalues
  
  for k = 1:K
    if strcmp(momtype,'dica')
      [theta,lam] = find_eigenpair(...
                 SX, W, K, M1, thetas, lams, precomputed, momtype);
    end
    if strcmp(momtype,'lda')
      [theta,lam] = find_eigenpair(...
                 SX, W, K, M1, thetas, lams, precomputed, momtype, c0, M2);
    end
    thetas(:,k) = theta;
    lams(k) = lam;
  end
  W = full(W);
  A = thetas'*W;
  
  Dest = estimate_D(A);
end

function [theta, lam] = find_eigenpair(...
                  SX, W, K, M1, thetas, lams, precomputed, momtype, c0, M2)
% lam is a robust eigenvalue
% theta is the respective robust eigenvector
  

  % It is known (theoretical result in the tpm paper) that for some bad
  % starting points tpm's convergence can be extremely slow. However, there
  % are other good starting points which converge fast and, in practice,
  % always to a better solution. Therefore, when looking for a robust
  % eigenpair, it is better to have several restarts (nruns) but limit the
  % number of iterations in each restart to some relatively small value
  % (maxiter). We experimentally found some reasonable values for these
  % parameters. There could be other good values.
  eps = 1e-5;
  maxiter = 100; % N in the tpm paper
  nruns = 10; % L in the tpm paper
  
  vects = cell(nruns,1);
  vals  = zeros(nruns,1);
  
  i = 1;
  while 1>0
    u = rand(K,1); u = u/norm(u);
    iter = 1;
    while 1>0
      if strcmp(momtype,'dica')
        WTW = compute_wtw_single_dica(SX, W, W'*u, M1, precomputed);
      end
      if strcmp(momtype,'lda')
        WTW = compute_wtw_single_lda(SX, W, W'*u, M1, M2, c0, precomputed);
      end
      gradient = WTW*u - thetas*(lams.*(thetas'*u).^2);
      unew = gradient/norm(gradient);
      
      if norm(u-unew) < eps, break, end
      u = unew;
      iter = iter + 1; 
      if iter > maxiter, break, end
    end
    
    vects{i} = u;
    vals(i) = u'*gradient;
    i = i + 1;
    
    if i > nruns, break, end
  end
  
  [~,ind] = max(vals);
  
  theta = vects{ind};
  lam = vals(ind);
  
end

function WTWs = compute_wtw_for_M_canonical_basis_dica(SX, W, K, M1)
%COMPUTE_WTW_FOR_M_CANONICAL_BASIS_DICA Computes M K-by-K matrices
%   W*T(e_1)*W', W*T(e_2)*W', ..., W*T(e_M)*W' where e_1, e_2, ..., e_M is
%   the canonical basis of R^M, i.e. the columns of the M-identity matrix,
%   and T is the DICA T-cumulant
%
% WTWs = compute_wtw_for_M_canonical_basis_dica(SX, W, K, M1)
%
% Input:
%   SX : sparse M-by-N matrix of word counts X with docs in columns
%   W  : K-by-M whitening matrix of the DICA S-cumulant
%   K  : number of topics
%   M1 : expectation of document's counts
%
% Output:
%   WTWs : K-by-(M*K) matrix [W*T(e_1)*W' W*T(e_2)*W' ... W*T(e_M)*W']
%
% Comment: M is the number of words in the dictionary and N is the
%   of documents in the corpus. Note that computing M projections of the
%   DICA T-cumulant is not necessary in practice although it would explain
%   all data contained in T. The joint diagonalization algorithm achieves
%   comparable results when run only, e.g., on K K-by-K matrices
%   W*T(W'*e_1)*W', W*T(W'*e_2)*W', ..., W*T(W'*e_K)*W' with e_1, e_2, ...,
%   e_K being the canonical basis of R^K. Therefore, this function is only
%   useful for either reproducing the results of our paper or for testing
%   whether your set of projection vectors is good enough.
%
% Warning: This function is provided for testing purpoces or for
%   reproducing the results of our paper. It is significantly slower than
%   other alternatives (without projecting onto the full basis of R^M).
%   This function is constructed to be used with jd_full_basis.m as opposed
%   to jd.m. While being computationally much faster, expected performance
%   of jd.m is approximately the same as that of jd_full_basis.m.

% Copyright 2015, Anastasia Podosinnikova

  [M, N] = size(SX);
  
  WTWs = zeros(K,K*M);  
  W    = sparse(W); 
  WX   = W*SX;
  WM1  = W*M1;
  temp = (2*N*WM1)*WM1' - WX*WX';
  SXt  = SX';
  
  XX = SX*SX';
  E = eye(M);
  
  for m = 1:M
    em  = sparse(E(:,m));
    Xm  = SXt*em;
    M1m = M1'*em;
    Wm  = W*em;
    
    temp0 = Wm*(WX*Xm)';
    temp1 = WM1*(WX*Xm)';
    temp2 = N*WM1*(M1m*Wm)';
    
    WTWs(:,K*(m-1)+1:K*m) ...
             = (N/((N-1)*(N-2))) * ( ...
                    WX*sparse(1:N,1:N,Xm)*WX' ...
                  + M1m * temp ...
                  - (temp1 + temp1')) ...
             + 2*M1m * (Wm * Wm') ...
             + (1/(N-1)) * ( ...
                  - temp0 - temp0' ...
                  + temp2 + temp2' ...
                  + W*sparse(1:M,1:M, N*M1m*M1 - XX(:,m))*W' ...
             );
  end
  
end

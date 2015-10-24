function WTWs = compute_wtw_for_M_canonical_basis_lda(SX, W, K, M1, M2, c0)
%COMPUTE_WTW_FOR_M_CANONICAL_BASIS_LDA Computes M K-by-K matrices
%   W*T(e_1)*W', W*T(e_2)*W', ..., W*T(e_M)*W' where e_1, e_2, ..., e_M is
%   the canonical basis of R^M, i.e. the columns of the M-identity matrix,
%   and T is the LDA T-moment
%
% WTWs = compute_wtw_for_M_canonical_basis_lda(SX, W, K, M1, M2, c0)
%
% Input:
%   SX : sparse M-by-N matrix of word counts X with docs in columns
%   W  : K-by-M whitening matrix of the LDA S-moment
%   K  : number of topics
%   M1 : the first LDA moment
%   M2 : the second LDA moment
%   c0 : parameter
%
% Output:
%   WTWs : K-by-(M*K) matrix [W*T(e_1)*W' W*T(e_2)*W' ... W*T(e_M)*W']
%
% Comment: M is the number of words in the dictionary and N is the
%   of documents in the corpus. Note that computing M projections of the
%   LDA T-moment is not necessary in practice although it would explain
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
  
  Ls = sum(SX)'; % % N-vector with lengths of each document
  delta3 = 1./( Ls.*(Ls-1).*(Ls-2) );
  
  W      = sparse(W); 
  WX     = W*SX;
  WXd3t  = (WX*sparse(1:N,1:N,delta3))';
  Xd3    = SX*delta3;
  WM2W   = W*M2*W';
  WM1    = W*M1;
  WM1WM1 = WM1*WM1';
  SXt    = SX';
  
  WTWs = zeros(K,K*M);
  E = eye(M);
  
  for m = 1:M
    em   = sparse(E(:,m));
    Xvd3 = (SXt*em).*delta3;
    M1v  = M1'*em;
    Wm   = W*em;
    Xm   = SXt*em;
    
    temp1 = Wm*(Xm' * WXd3t);
    temp2 = (W*(M2*em))*WM1';
    
    WTWs(:,K*(m-1)+1:K*m) ...
                = (1/N) * ( ...
                        + WX*sparse(1:N,1:N,Xvd3)*WX' ...
                        + (2*Xd3(m)*Wm)*Wm' ...
                        - W*sparse(1:M,1:M,SX*Xvd3)*W' ...
                        - temp1 - temp1') ...
                - (c0/(c0+2)) * (M1v*WM2W + temp2 + temp2') ...
                + (2*c0^2/((c0+1)*(c0+2))) * (M1v*WM1WM1);   
  end
  
end

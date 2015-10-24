function WTWs = compute_multiple_wtw_lda(SX, W, K, vecs, M1, M2, c0)
%COMPUTE_MULTIPLE_WTW_LDA Computes P K-by-K matrices W*T(v_1)*W', 
%   W*T(v_2)*W', ..., W*T(v_P)*W' for the LDA T-moments and P vectors 
%   v_1, v_2, ..., v_P
%
% WTWs = compute_multiple_wtw_lda(SX, W, K, vecs, M1, M2, c0)
%
% Input: 
%   SX   : sparse M-by-N matrix of word counts X with docs in columns
%   W    : K-by-M whitening matrix of the LDA S-moments
%   K    : number of topics
%   vecs : cell array {v_1, v_2, ..., v_P}
%   M1   : the first LDA moment
%   M2   : the second LDA moment
%   c0   : parameter (reasonable values are, e.g., of the order 0.01)
%
% Output:
%   WTWs : K-by-(P*K) matrix [W*T(v_1)*W' W*T(v_2)*W' ... W*T(v_P)*W']
%
% Comment: M is the number of words in the dictionary, N is the number
%   of documents in the corpus, and P is the number of projections, i.e.
%   the number of vectors v_1, v_2, ..., v_P. Note that in most cases it is
%   beneficial to set each vector v_p to W'*u_p, where u_p is a K-vector. 
%   It is often sufficient to set P = K and vectors u_p to the canonical 
%   basis of R^K, i.e. the colums of the K-identity matrix. In the LDA 
%   model, if the topic mixture theta ~ Dirichlet(c) with c being K-vector,
%   then the parameter c0 = sum_{k=1^K} c_k.

% Copyright 2015, Anastasia Podosinnikova
  
  P = length(vecs);
  W = sparse(W);
  precomputed = precompute_for_wtw(SX, W, 'lda', M1, M2);
  
  WTWs = zeros(K,K*P);
  for p=1:P
    v = vecs{p};
    WTWs(:,K*(p-1)+1:K*p) = compute_wtw_single_lda(SX,W,v,M1,M2,c0,precomputed);
  end
  
end
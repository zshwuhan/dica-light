function [W, M1, M2, VK, DK] = compute_S_and_W_lda(SX, K, c0)
%COMPUTE_S_AND_W_LDA Computes LDA S-moment and its whitening matrix W
%
% [W, S, VK, DK, M1, M2] = compute_S_and_W_lda(SX, K, c0)
%
% Input:
%   SX : sparse M-by-N matrix of word counts X with docs in columns
%   K  : number of topics
%   c0 : parameter (reasonable values are, e.g., of the order 0.01)
%
% Output:
%   W  : K-by-M whitening matrix of S
%   M1 : the first LDA moment
%   M2 : the second LDA moment
%
% Comment: M is the numebr of words in the dictionary and N is the
%   number of documents in the corpus. In the LDA model, if the topic
%   mixture theta ~ Dirichlet(c) with c being K-vector, then the parameter
%   c0 = sum_{k=1^K} c_k.

% Copyright 2015, Anastasia Podosinnikova

  [M, N] = size(SX);
  
  Ls = sum(SX)'; % N-vector with lengths of each document
  if sum(find(Ls<3))>0
    error('To compute LDA moments, each document has to have at least 3 tokens') 
  end
  
  delta1 = 1./Ls;
  delta2 = 1./(Ls.*(Ls-1));
  
  M1 = (SX*delta1) / N;
  M2 = (1/N)*(SX*sparse(1:N,1:N,delta2)*SX' - sparse(1:M,1:M,SX*delta2));
  M2 = full(M2);
  S = M2 - ((c0/(c0+1))*M1)*M1';
  [W, VK, DK] = compute_whitening_matrix(S, K);
    
end

% MODIFICATION If S is needed (note, it is M-by-M matrix!) then modify the
% first line as follows
%   [W, M1, M2, S] = compute_S_and_W_lda(SX, K, c0)

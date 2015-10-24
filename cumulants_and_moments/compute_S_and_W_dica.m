function [W, M1, VK, DK] = compute_S_and_W_dica(SX, K)
%COMPUTE_S_AND_W_DICA Computes DICA S-cumulant and its whitening matrix W
%
% [W, M1] = compute_S_and_W_dica(SX, K)
%
% Input:
%   SX : sparse M-by-N matrix of word counts X with docs in columns
%   K  : number of topics
%
% Output:
%   W  : K-by-M whitening matrix of S
%   M1 : expectation of document's counts
%
% Comment: M is the number of words in the dictionary and N is the
%   number of documents in the corpus.

% Copyright 2015, Anastasia Podosinnikova

  [M, N] = size(SX);
  M1 = sum(SX,2)/N;
  covx = ((1/(N-1))*SX)*SX' - ((N/(N-1))*M1)*M1';
  S = covx - sparse(1:M,1:M,M1);
  [W, VK, DK] = compute_whitening_matrix(full(S), K);
    
end

% MODIFICATION If S is needed (note, it is M-by-M matrix!) then modify the
% first line as follows
%   [W, M1, S] = compute_S_and_W_dica(SX, K)

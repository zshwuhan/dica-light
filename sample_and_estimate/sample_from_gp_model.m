function [SX, Alphs] = sample_from_gp_model(K, M, N, D, c, b)
%SAMPLE_FROM_GP_MODEL Samples documents and respective hidden variables
%   from the GP model
%
% [SX, Alphs] = sample_from_gp_model(K, M, N, D, c, b)
%
% Input:
%   K    : number of topics
%   M    : number of words in the dictionary
%   N    : number of documents
%   D    : M-by-K topic matrix
%   c, b : parameters of the gamma distribution (as defined in our paper)
%
% Output:
%   SX     : M-by-N sparse matrix, its n-th column is the count vector of
%            the n-th document
%   Alphas : K-by-N matrix, its n-th column is the alpha vector of the n-th
%            document

% Copyright 2015, Anastasia Podosinnikova
    
  if size(D,1) ~= M || size(D,2) ~= K, error('Wrong input'), end

  Alphs = zeros(K,N);
  SX = sparse(M,N);
  if isrow(c), c = c'; end
  
  % sample in batches
  n = 1000;
  times = floor(N/n);
  rest = mod(N,n);
  
  for i=1:times
    inds = (i-1)*n+1:i*n;
    [alphs,sx] = sample_batch(D,c,b,K,n);
    Alphs(:,inds) = alphs;
    SX(:,inds) = sx;
  end
  
  inds = times*n + 1 : times*n + rest;
  [alphs,sx] = sample_batch(D,c,b,K,rest);
  Alphs(:,inds) = alphs;
  SX(:,inds) = sx;
    
end

function [alphs,sx] = sample_batch(D,c,b,K,n)
  alphs = gamrnd(repmat(c,1,n),repmat(1/b,K,n));
  sx = sparse(poissrnd(D*alphs));
end

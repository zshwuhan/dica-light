function D = sample_toy_topic_matrix(M, K, a)
%SAMPLE_TOY_TOPIC_MATRIX Samples an M-by-K matrix with each column from the
%   symmetric Dirichlet distribution with the concentration parameter a
%
% D = sample_toy_topic_matrix(M, K, a)
%
% Input:
%   M : number of words in the dictionary
%   K : number of topics
%   a : concentration parameter
%
% Output:
%   D : sampled toy M-by-K topic matrix

% Copyright 2015, Anastasia Podosinnikova

  basemeasure = ones(M, 1);
  param = a*basemeasure;
  encore = 1; iter = 1;
  while encore
    temp = gamrnd(repmat(param,1,K), 1, M, K);
    r = sum(temp, 1);
    r(logical(r == 0)) = 1;
    D = temp./repmat(r, M, 1);
    sss = svd(D);
    % ensures that the sampled matrix is well conditioned
    if sss(1)/sss(K) <= 15, encore = 0; end
    iter = iter + 1;
    if iter > 100, error('Try different parameters'), end
  end

end

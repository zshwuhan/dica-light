function [err1, perm] = l1_error(Dest, D)
%L1_ERROR Computes the L1-error between two topic matrices Dest and D of
%   the same size
%
% [err1, rp] = l1_error(Dest, D)
%
% Input:
%   Dest : M-by-K topic matrix estimated by an algorithm
%   D    : M-by-K ground truth topic matrix
%
% Output:
%   err1 : the estimated error scaled to the [0,1] interval
%   perm : permutation of the columns of Dest corresponding to the
%          estimated error err1 (Dest(:,perm) \approx D) when properly
%          renormalized

% Copyright 2015, Anastasia Podosinnikova

  [Dest, D, K] = verify_correctness_of_the_input_and_normalize(Dest, D);
  
  Perf = zeros(K);
  for i=1:K, 
    for j=1:K, 
      Perf(i,j) = norm(Dest(:,i)-D(:,j), 1); % l1-norm!
    end
  end
  
  % as the columns of the topic matrix are estimated up to permutations
  [matching, cost] = HungarianBipartiteMatching(Perf);
  
  [perm, ~] = find(sparse(matching));
  err1 = cost/K;
  err1 = err1/2; % to scale in [0,1]
  
end

function [Dest, D, K] = verify_correctness_of_the_input_and_normalize(Dest, D)

  if (size(Dest,1) ~= size(D,1)) || (size(Dest,2) ~= size(D,2))
    error('Wrong input size')
  end

  if (sum(sum(isnan(D))) > 0) || (sum(sum(isnan(Dest))) > 0)
    error('Wrong input')
  end
  
  [M, K] = size(D);
  
  if (sum(sum(find(Dest<0))) > 0) || (sum(sum(find(D<0))) > 0)
    % check non-negativity
    error('Wrong input')
  end
  
  if (abs(sum(sum(Dest)) - K) > 1e-8) || (abs(sum(sum(D)) - K) > 1e-8)
    % check the simplex constraint
    warning('Matrices were renormalized')
    Dest = Dest ./ repmat(sum(Dest), M, 1);
    D = D ./ repmat(sum(D), M, 1);
  end
  
end

function [D, signs] = estimate_D(A)
%ESTIMATE_D Estimates topic matrix from the output of spectral, jd, or tpm
%
% D = estimate_D(A);
%
% Input:
%   A : output of spectral, jd, or tpm
%
% Output:
%   D     : estimate of the topic matrix
%   signs : signs assigned to the columns of pinv(A)
%
% Comment: Just taking the pseudo-inverse of A is not sufficient because it
%   can (and in practice does) introduce negative values. This is due to
%   the noise present in the data. Indeed, if the data is infinite and comes
%   from the correct model (LDA or DICA) then no negative values occur. 
%   In practice, however, one has to deal with the noise due to finite 
%   samples and deviations from the correct model. Therefore, some
%   heuristic post-processing to eliminate negative values is required.

% Copyright 2015, Anastasia Podosinnikova

  M = size(A,2);
  
  % problem: the pseudo inverse can introduce negative values
  Dest = pinv(A);
  
  % each column of Dest is estimated up to multiplication by scalar
  % => checking wheter columns have correct signs
  [Dest, signs] = flip_column_signs(Dest);
  
  % truncate all negative values
  Dest = max(0,Dest);
  
  % normalize each column to be in the simplex
  D = Dest./repmat(sum(Dest),M,1);
  
end

function [Dest, signs] = flip_column_signs(Dest)
  K = size(Dest,2);
  signs = ones(1,K);
  % if in the column more negative values than positive => switch the sign
  % can be implemented in many different ways
  for k = 1:K
    val1 = sum(min(0,Dest(:,k)).^2);
    val2 = sum(max(0,Dest(:,k)).^2);
    signs(k) = sign(val2 - val1);
  end
  Dest = Dest*diag(signs);
end

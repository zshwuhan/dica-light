function [V, diags] = joint_diagonalization(B, varargin)
%JOINT_DIAGONALIZATION Joint diagonalization routine
%
% [V, diags] = joint_diagonalization(B)
% [V, diags] = joint_diagonalization(B, isrand)
% 
% Input: 
%     B      : m-by-(n*m) matrix of n m-by-m matrices to be jointly
%              diagonalized
%     isrand : (optional) 
%              - if set to 1, V0 is initialized with a random orthogonal 
%                 m-by-m matrix
%              - if set to 0, V0 is initialized with the m-by-m identity
%                matrix (default)
%
% Output: 
%     V     : orthogonal matrix, the output of the joint diagonalization
%     diags : cell array of (almost) diagonal matrices
%
% Comment: This function is for calling the C++/MEX-Matlab function
%   "jd_in.cpp". See its comments for details.

% COPYRIGHT 2015, Anastasia Podosinnikova

  % n = the number of the matrices to be jointly diagonalized
  % size of each matrix (m-by-m)
  [m, nm] = size(B);
  n = nm/m; if m*n~=nm, error('Wrong input size'), end

  % set parameters
  eps = 1e-8;
  V0 = eye(m);
  if length(varargin)==1
    isrand = varargin{1};
    if isrand == 1
      V0 = sample_orthogonal_matrix(m);
    end
  end
  
  % run jd_in.cpp
  [out1, out2] = jd_in( B(:), m, n, eps, V0(:)); 
  
  [V, diags] = reshape_the_output_of_jd_in_cpp(out1, out2, m, n);
  
end

function [V, diags] = reshape_the_output_of_jd_in_cpp(out1, out2, m, n)
% reshape the output of jd_in.cpp
  V = reshape(out1, m, m);
  diags_temp = reshape(out2, m, m*n);
  diags = cell(1,n);
  for i = 1:n, 
    diags{i} = diags_temp(:, (i-1)*m+1:i*m); 
  end
end

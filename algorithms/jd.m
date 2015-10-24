function [Dest, cmed, cmean, A, as, W, vecs] = jd(SX, K, momtype, opts, c0)
%JD Joint diagonalization for topic models
%
% [A, diags, W, vecs] = jd(SX, K, momtype, opts, c0)
%
% Input:
%   SX      : sparse M-by-N matrix of word counts X with docs in columns
%   K       : number of topics
%   momtype : either 'lda' for LDA moments or 'dica' for DICA cumulants
%   opts    : either 0 (for defaults) 
%             
%             or struct with the following fields:
%             * 'type' defines the way projection vectors are generated;
%               - if set to 'default', K projection vectors W'*e_1, W'*e_2,
%                 ..., W'*e_P are sampled where e_1, e_2, ..., e_P is the
%                 canonical basis of R^K (the columns of the K-identity
%                 matrix) (default)
%               - if set to 'random' then another filed opts.P have to be
%                 given where P is the number of the projection vectors and
%                 each of them is sampled as W'*u where u is a vector 
%                 sampled uniformly at random from the unit K-sphere
%             * 'P' is the number of projection vectors if type = 'random'
%             * 'isrand' is the initialization type for the algorithm
%               - if set to 0, it is initialized with the K-identity matrix
%                 (default)
%               - if set to 1, it is initialized with a random K-by-K
%                 orthogonal matrix
%             * 'nruns' is the number of repetitions if isrand = 1; has to
%                 greater or equal to 1; the algorithm with random
%                 initialization is run 'nruns' times and the output with 
%                 the best (smallest) objective is reported
%
%   c0      : parameter for LDA moments (only if momtype = 'lda')
%
% Output:
%   Dest : estimate of the parameter D (topic matrix)
%   c    : estimate of the parameter c
%   A    : K-by-M diagonalizing matrix
%   as   : cell array with almost diagonal matrices A*S*A', A*T(v_p)*A'
%   W    : K-by-M whitening matrix
%   vecs : the vectors, which were used for the projections
%   
% Comment 1: See "estimate_D.m" for estimation of the topic matrix.
%
% Comment 2: Computing M projections of the LDA T-moment or the
%   DICA T-cumulant is not necessary in practice although it would explain
%   all the  data contained in the tensor T. In practice, the joint
%   diagonalization algorithm achieves comparable (with the version based
%   on M projections) results when run only, e.g., on K K-by-K matrices 
%   W*T(W'*e_1)*W', W*T(W'*e_2)*W', ..., W*T(W'*e_K)*W' with e_1, e_2, ...,
%   e_K being the canonical basis of R^K. For the version whith M 
%   projections computed, check the "jd_full_basis.m" implementation.

% Copyright 2015, Anastasia Podosinnikova


  opts = verify_correctness_of_the_input_and_set_defaults(momtype,opts);

  % compute a whitening matrix
  if strcmp(momtype,'dica')
    [W, M1] = compute_S_and_W_dica(SX, K); 
  end
  if strcmp(momtype,'lda')
    [W, M1, M2] = compute_S_and_W_lda(SX, K, c0); 
  end
  
  % construction of projection vectors is whitening matrix dependent
  if strcmp(opts.type,'default')
    E = eye(K);
    vecs = cell(K,1);
    for k = 1:K, vecs{k} = W'*E(:,k); end
  end
  if strcmp(opts.type,'random')
    P = opts.P;
    vecs = cell(P,1);
    for k = 1:P, u = rand(K,1); vecs{k} = W'*(u/norm(u)); end
  end
  
  % compute W*T(v)*W's
  if strcmp(momtype,'dica')
    WTWs = compute_multiple_wtw_dica(SX, W, K, vecs, M1);
  end
  if strcmp(momtype,'lda')
    WTWs = compute_multiple_wtw_lda(SX, W, K, vecs, M1, M2, c0);
  end
  
  % perform joint diagonalization
  if opts.isrand == 0
    [V, as] = joint_diagonalization([eye(K) WTWs], opts.isrand); % (!) eye(K) = W*S*W'
  end
  
  if opts.isrand == 1
    Vs = cell(opts.nruns,1);
    Ds = cell(opts.nruns,1);
    objs = zeros(opts.nruns,1);
    for irun = 1:opts.nruns
      [V, as] = joint_diagonalization([eye(K) WTWs], opts.isrand); % (!) eye(K) = W*S*W'
      obj = compute_jd_objective([eye(K) WTWs], V);
      Vs{irun} = V; Ds{irun} = as; objs(irun) = obj;
    end
    [~,ind] = min(objs);
    V = Vs{ind};
    as = Ds{ind};
  end
  
  A = V'*W;
  
  
  
  % Estimation of the parameters (D, c)
  
  M = size(A,2);
  % problem: the pseudo inverse can introduce negative values
  Dest = pinv(A);
  % each column of Dest is estimated up to multiplication by scalar
  % => checking wheter columns have correct signs
  [Dest, signs] = flip_column_signs(Dest);
  % truncate all negative values
  Dest = max(0,Dest);
  
  % before normalizing Dest, compute c
  P = size(vecs,1); 
  C = zeros(K,P);
  for k = 1:K
    for p = 1:P
      a = as{p};
      if strcmp(momtype,'dica')
        C(k,p) = 4*(Dest(:,k)'*signs(k)*vecs{p})^2 / a(k,k)^2;
      end
      if strcmp(momtype,'lda')
        C(k,p) = 4*(Dest(:,k)'*signs(k)*vecs{p})^2 / a(k,k)^2;
      end
    end
    if strcmp(momtype,'lda')
      C = C * c0*(c0+1)/(c0+2)^2;
    end
  end
  cmean = sum(C,2) / P;
  cmed = zeros(K,1);
  for k = 1:k
    [~,ind] = sort(C(k,:),'ascend');
    cmed(k) = C(k,ind(ceil(K/2)));
  end
  
  % normalize each column of Dest to be in the simplex
  Dest = Dest./repmat(sum(Dest),M,1);
  
end



function opts = verify_correctness_of_the_input_and_set_defaults(momtype,opts)

  if ~( strcmp(momtype,'dica') || strcmp(momtype,'lda') )
    error('Wrong momtype')
  end
  
  if ~isstruct(opts), 
    opts = struct();
  end
  
  if ~isfield(opts,'type')
    opts(1).type = 'default';
  end
  
  if strcmp(opts.type,'random')
    if ~isfield(opts,'P')
      error('for type random, specify the number of projection vectors P')
    end
  end
  
  if ~( strcmp(opts.type,'default') || strcmp(opts.type,'random') )
    error('Wrong opts.type')
  end
  
  if ~isfield(opts,'isrand')
    opts(1).isrand = 0;
  end
  
  if ~( opts.isrand == 1 || opts.isrand == 0 )
    error('Wrong opts.isrand')
  end
  
  if opts.isrand == 1
    if ~isfield(opts,'nruns')
      error('for isrand = 1, specify the number of random restarts nruns')
    end
    if opts.nruns < 1
      error('nruns have to be a positive integer number: 1,2,3,...')
    end
  end
  
end


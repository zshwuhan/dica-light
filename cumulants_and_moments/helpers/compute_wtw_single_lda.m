function WTW = compute_wtw_single_lda(SX, W, v, M1, M2, c0, precomputed)
% COMPUTE_WTW_SINGLE_LDA Computes W*T(v)*W' for the LDA T-moment
%
% WTW = compute_wtw_single_lda(SX, W, v, M1, M2, c0, precomputed)
%
% INPUT:
%   SX : sparse M-by-N matrix of word counts X with docs in columns
%   W  : K-by-M whitening matrix of the LDA S-moment
%   v  : M-vector, in practice, W'*u where u is a K-vector
%   M1 : the first LDA moment
%   M2 : the second LDA moment
%   c0 : parameter (reasonable values are, e.g., of the order 0.01)
%   precomputed : some precomputed values (see precompute_for_wtw.m)
%
% OUTPUT:
%   WTW : K-by-K matrix W*T(v)*W'
%
% COMMENTS: This function computes W*T(v)*W', where T is the LDA
%   T-moment, W is a whitening matrix of the LDA S-moment, and v is a
%   vector. When computing several K-by-K matrices W*T(v_1)*W', 
%   W*T(v_2)*W', ..., W*T(v_P)*W', it is sufficient to perform some 
%   computations only ones instead of P times. The result of these 
%   computations is contained in the variable called precomputed. M is the 
%   number of words in the dictionary, N is the number of documents in the 
%   corpus, and K is the number of topics. In the LDA model, if the topic
%   mixture theta ~ Dirichlet(c) with c being K-vector, then the parameter
%   c0 = sum_{k=1^K} c_k.

% Copyright 2015, Anastasia Podosinnikova - INRIA - ENS

  [M, N] = size(SX);
  
  delta3 = precomputed.delta3;
  WX     = precomputed.WX;
  WXd3t  = precomputed.WXd3t;
  Xd3    = precomputed.Xd3;
  WM2W   = precomputed.WM2W;
  WM1    = precomputed.WM1;
  WM1WM1 = precomputed.WM1WM1;
  
  Xvd3 = (SX'*v).*delta3;
  M1v  = M1'*v;

  temp1 = (W*sparse(1:M,1:M,v)*SX) * WXd3t;
  temp2 = (W*(M2*v))*WM1';

  WTW ...
    = (1/N) * ( ...
            + WX*sparse(1:N,1:N,Xvd3)*WX' ...
            + W*sparse(1:M,1:M,2*(Xd3.*v) - SX*Xvd3)*W' ...
            - temp1 - temp1') ...
    - (c0/(c0+2)) * (M1v*WM2W + temp2 + temp2') ...
    + (2*c0^2/((c0+1)*(c0+2))) * (M1v*WM1WM1);
        
end

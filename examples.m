%EXAMPLES Examples of use of the moment matching topic modeling algorithms.

% Copyright 2015, Anastasia Podosinnikova


% STEP 1: sample data

M = 50; % number of words in the dictionary
N = 1000; % number of documents in the corpus
K = 5; % number of topics
a = 0.1; % sparsity parameter for a toy topic matrix
D = sample_toy_topic_matrix(M, K, a);

c = 0.1*ones(K,1); b = 0.1;
SX = sample_from_gp_model(K, M, N, D, c, b);
inds = find(sum(SX)<3); % the LDA moments require at least 3 documents
SX = SX(:,setdiff(1:N,inds));


% STEP 2: set parameters

c0 = 0.1*K; % for the LDA moments (this is the true value!)


% STEP 3: run algorithms

% (1) spectral for the DICA cumulants
momtype = 'dica';
tic, Dest = spectral(SX, K, momtype); time = toc;
err1 = l1_error(Dest, D);
disp(['spectral (dica) l1-error = ',num2str(round(err1*1000)/1000),',   time = ',num2str(time),' sec'])

% (2) spectral for the LDA moments
momtype = 'lda';
tic, Dest = spectral(SX, K, momtype, c0); time = toc;
err1 = l1_error(Dest, D);
disp(['spectral (lda)  l1-error = ',num2str(round(err1*1000)/1000),',   time = ',num2str(time),' sec'])

% (3) jd for the DICA cumulants
momtype = 'dica';
opts = 0; % default
%opts = struct('type','default','isrand',0); %equivalent to default
%opts = struct('type','default','isrand',1);
%opts = struct('type','random','P',2*K,'isrand',0);
%opts = struct('type','random','P',2*K,'isrand',1);
tic, Dest = jd(SX, K, momtype, opts); time = toc;
err1 = l1_error(Dest, D);
disp(['jd (dica)       l1-error = ',num2str(round(err1*1000)/1000),',   time = ',num2str(time),' sec'])

% (4) jd for the LDA moments
momtype = 'lda';
opts = 0; % default
tic, Dest = jd(SX, K, momtype, opts, c0); time = toc;
err1 = l1_error(Dest, D);
disp(['jd (lda)        l1-error = ',num2str(round(err1*1000)/1000),',   time = ',num2str(time),' sec'])

% (5) tpm for the DICA cumulants
momtype = 'dica';
tic, Dest = tpm(SX, K, momtype); time = toc;
err1 = l1_error(Dest, D);
disp(['tpm (dica)      l1-error = ',num2str(round(err1*1000)/1000),',   time = ',num2str(time),' sec'])

% (6) tpm for the LDA moments
momtype = 'lda';
tic, Dest = tpm(SX, K, momtype, c0); time = toc;
err1 = l1_error(Dest, D);
disp(['tpm (lda)       l1-error = ',num2str(round(err1*1000)/1000),',   time = ',num2str(time),' sec'])


disp('Note that this script only demonstrates the usage of the algorithms.')
disp('Do not draw too many conclustions from these toy data.')

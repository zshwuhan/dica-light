%INSTALL Adds all required paths and builds MEX-files

addpath(pwd);

cd algorithms; addpath(pwd); cd ..
cd algorithms/helpers; addpath(pwd); cd ../..
cd cumulants_and_moments; addpath(pwd); cd ..
cd cumulants_and_moments/helpers; addpath(pwd); cd ../..
cd sample_and_estimate; addpath(pwd); cd ..

cd algorithms/helpers; make_jd; cd ../..

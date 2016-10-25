%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                        %%
%%   TEST ROUTINE FOR PHASE RETRIEVAL SOLVED BY ADMM-TV   %% 
%%                                                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%  Clear/close everything
clear all;
close all;
fprintf( '\nTEST ROUTINE FOR PHASE RETRIEVAL SOLVED BY ADMM-TV\n' );

%%  Load flat field intensity map phantom

phantom_name='test_phantom.mat';

%% Load detector intensity
load(phantom_name,'img');
[n,m] = size( img );

%%  Load mask
load(phantom_name,'mask');
scirc = strel('square',2);
mask = imdilate(mask, scirc);

%% Physical Parameters
E=17.01;%keV
lambda=12.4/E*1e-10;
pxs=1e-8;% pixelsize
DOF=pxs^2/lambda;
D=[DOF*100];
betaoverdelta=5e-1;

%%  ADMM-TV setting
niter  = 200;                          %%  number of iterations
eps    = 1e-3;                         %%  stopping threshold
tau   = 5e-5;                          %%  connection strength
eta = 0.02*tau;                        %%  regularization strength
%
phys   = 0;                            %%  flag for the physical constraints
ks     = size( kernel_grad , 1 )-1;    %%  size of the gradient kernel

fprintf( '\nADMM setting:' );
fprintf( '\nniter max = %d' , niter );
fprintf( '\neps = %.3e' , eps );
fprintf( '\neta = %.3e' , eta );
%fprintf( '\nlambda2 = %.3e' , lambda2 );
fprintf( '\ntau = %.3e' , tau );
fprintf( '\nks = %d\n' , ks );

%%Padding image 
b = padarray(img,[ks,ks],'replicate');

%% FPSF (Fourier transformed of the PSF)

FPSF=[];% If FPSF or OTF is required, define it here

%%  Display input data
figure;  imagesc( b );  colormap gray; axis off; 
title( 'Input intensity' );
if mean( mask(:) ) ~= 1.0 
    figure;  imagesc( mask );  colormap gray;  title( 'Support constraint' );
end

%%  Iterative reconstruction

fprintf('\n\nStarting reconstruction with ADMM ....');
tic;
x_it = admm_ctf_betaoverdelta( b , niter , eps , eta, tau ,phys , mask, lambda,D, pxs,betaoverdelta, FPSF );
TimeInterval = toc;
fprintf('\n.... reconstruction done!\n');

figure;  imagesc( x_it );  colormap gray;axis off; title( 'Iterative reconstruction' );

%%  Analytical reconstruction

epsilon=1e-1;
x_an = ctf_fixedratio_retrieval( b,epsilon,D,lambda,pxs,betaoverdelta,FPSF );  
x_an = x_an(ks+1:n+ks,ks+1:m+ks);
figure;  imagesc( x_an ); axis off;  colormap gray;  title( 'Analytical reconstruction' );


%%  Compute mean squared error

load(phantom_name,'Uin');% Load oracle
phase_map=angle(Uin);
figure;  imagesc(phase_map );  axis off; colormap gray;  
title( 'Phantom Phase map' );

err_it = psnr( x_it , phase_map , 1 );
err_an = psnr( x_an , phase_map , 1 );
fprintf( '\n\nIterative reconstruction error (PSNR) : %.5e' , err_it );
fprintf( '\nAnalytical reconstruction error (PSNR): %.5e\n' , err_an );
fprintf( 'Time elapsed: %.5f sec.\n\n' , TimeInterval );



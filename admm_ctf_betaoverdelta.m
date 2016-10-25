%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                       %%  
%%   Alternate direction method of multipliers solving   %% 
%%              LASSO-TV for phase retrieval             %%
%%                                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%  The ADMM minimizes the following augmented Lagrangian:
%%
%%  Lagr = 1/2 || Hx - b ||^{2}_{2} + lambda * sum_{i} || u_{i} ||_{1} +
%%         - alpha^{T} ( u - Gx ) + beta/2 || u - Gx ||^{2}_{2}
%%
%%  x     --->  pure phase object to retrieve
%%  H     --->  CTF operator assuming beta over delta
%%  b     --->  input data
%%  u     --->  dual variable
%%  lambda1, beta --->  multipliers
%%  G     --->  gradient operator
%% lambda ---> wavelength
%% z      ---> propagation distance
%% pxs    ---> pixel size of the detector
%% betaoverdelta    ---> refractive index ratio between beta and delta


function x = admm_ctf_betaoverdelta( b , niter , eps , lambda1, beta , phys , mask, lambda,z, pxs,betaoverdelta,OTF )
    %%  Define operators

    H   = @operator_ctf_deltabetaphaseretrieval;%ctf_phase_operator;
    G   = @grad;
    Gt  = @grad_adj;
    IBT = @inv_block_toeplitz_ctf_betaoverdelta;
    % Regularization
    b=b-1;
    
    %%  Allocate memory for auxiliary arrays
    ks      = size( kernel_grad() , 1 );
    [n,m]   = size( b );

    n1       = n - 2*( ks - 1 );
    m1       = m - 2*( ks - 1 );
    n2       = n1 + ( ks - 1 );
    m2       = m1 + ( ks - 1 );
    n3       = n1 + 2*( ks - 1 );
    m3       = m1 + 2*( ks - 1 );
    na       = round( 0.5 * ( n3 - n1 ) );
    ma       = round( 0.5 * ( m3 - m1 ) );
    
    x       = zeros( n1 , m1 );
    aux     = zeros( n3 , m3 );
    xold    = zeros( n1 , m1 );
    u       = zeros( n2 , m2 , 2 );
    alpha   = zeros( n2 , m2 , 2 );
    obj_old = 1e20;

    %%  Start loop
    for it = 1:niter
        
        %%  Solve x subproblem        
        xold(:,:) = x;
        aux(:,:)  = IBT( H( b,z,lambda,pxs,betaoverdelta,OTF) + Gt(beta * u - alpha ) , beta , z, lambda,pxs, betaoverdelta); 
        x(:,:)    = aux(na+1:na+n1,ma+1:ma+m1);
        
        %%  Enforce support and physical constraints
        if mean( mask(:) ) ~= 1.0 
            zero_value=mean(x(mask == 0));
            x=x-zero_value;
            %x( mask == 0 ) = x( mask==0 )-0.2*xold( mask==0 ); %HIO correction
        end
        if phys == 1
            x( x < 0 ) = 0.0;
        end
        
        
        %%  Solve u subproblem
        u(:,:,:) = G( x ) + 1.0/beta * alpha;
        u(:,:,:) = shrinkage( u , lambda1/beta );
        
        
        %%  Update multipliers
        alpha(:,:,:) = alpha + beta * ( G( x ) - u );
        
        
        %%  Display current reconstruction
        %figure;  imagesc( x );  colormap gray;  title( strcat( 'Iteration n.' , num2str( it ) ) ); 
        
        
        %%  Compute iteration error
        err = error_intensity( x , xold , z,lambda,pxs,betaoverdelta,OTF);
        %err = norm( x - xold);
        obj = objective( x , b , lambda1 , u ,z,lambda,pxs);
        fprintf( '\n\tADMM iteration n.%d  --->  err. rel.: %.5e    obj.:%.5e' , it , err , obj );
        
        if err < eps
            fprintf( '\nADMM stopped by error criterion\n' );
            break
        end
%         if obj > obj_old
%             fprintf( '\nADMM stopped for increase of the functional\n' );
%             break;
%         else
            obj_old = obj;
%         end        
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                       %%  
%%                     Shrinkage operator                %%
%%                                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function u = shrinkage( u , kappa )
    u(:,:,:) = max( 0 , u - kappa ) - max( 0 , -u - kappa );
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                                       %%  
%%                        Objective                      %%
%%                                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function obj = objective( x , b , lambda1 , u, D, lambda, pxs, betaoverdelta )
    [n,m] = size( x );
    z = sqrt( u(:,:,1).^2 + u(:,:,2).^2 );
    ks     = size( kernel_grad , 1 )-1;
    aux= padarray(x,[ks,ks],'replicate');
    %aux=operator_ctf_deltabetaphaseretrieval( aux,D,lambda,pxs,betaoverdelta);
    obj = 0.5 * norm( aux  - b ).^2 + lambda * norm( z(1:n,1:m) , 1 );
end

function err = error_intensity( x , x_old, D,lambda,pxs,betaoverdelta,OTF )
    ks     = size( kernel_grad , 1 )-1;
    x= padarray(x,[ks,ks],'replicate');
    x_old= padarray(x_old,[ks,ks],'replicate');
    [n,m]=size(x);
    aux= operator_ctf_deltabetaphaseretrieval( x,D,lambda,pxs, ...
                                               betaoverdelta,OTF);
    aux_old= operator_ctf_deltabetaphaseretrieval( x_old,D,lambda,pxs, ...
                                               betaoverdelta,OTF);
    err = norm( aux_old - aux );
end


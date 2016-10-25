%%  Gradient operator performed through convolution

function g = grad( x )
    kx = kernel_grad();
    ks = size( kx , 1 );
    ky = kx';
    
    [ n , m ] = size( x );
    g = zeros( n + ks - 1 , m + ks - 1 , 2 );
    
    g(:,:,1) = conv2( x , kx);
    g(:,:,2) = conv2( x , ky);
end

%%  Gradient adjoint operator performed through convolution

function x = grad_adj( g )
    kx = kernel_grad();
    ky = kx';
    
    x1 = conv2( g(:,:,1) , kx );
    x2 = conv2( g(:,:,2) , ky );
    x  = - ( x1 + x2 );
end

%%  Invert block-toeplitz operator ( H^{2} + beta * Gt * G  ) through FFT

function x = inv_block_toeplitz_ctf_betaoverdelta( b , beta , z, lambda,pxs,betaoverdelta ,OTF )
    if nargin==6
        OTF=[];
    end
    % Kernel H operator
    [ n1 , n2 ] = size( b );
    f1=ifftshift([-fix(n2/2):ceil(n2/2)-1])/n2;
    f2=ifftshift([-fix(n1/2):ceil(n1/2)-1])/n1;
    [f1,f2]=meshgrid(f1,f2);
    fmap = f1.^2 + f2.^2;
    clear f1 f2; 
    sinfmap=sin(pi*lambda*z*fmap/pxs^2);
    cosfmap=cos(pi*lambda*z*fmap/pxs^2);
    numerator=2*(sinfmap-betaoverdelta*cosfmap);
    if ~isempty(OTF)
        numerator=numerator.*OTF;
        %numerator=numerator.*fft2(ifftshift(PSF));
    end
    kkl=numerator.^2;    
    % Kernel gradient part
    kx   = kernel_grad();
    kkx  = conv2( kx , kx );
    ky   = kx';
    kky  = conv2( ky , ky );
    kk = zeros( size( b ) );
    kk(1:size(kkx,1),1:size(kkx,2)) = -kkx-kky;
    % Alignment with the other image
    kk = fft2(circshift( kk , - [ 2 2 ] ) );
    %
    S = kkl+ beta*kk+1e-14;
    x = real( fftshift(ifft2( fft2( ifftshift(b )) ./ S ) ));
end


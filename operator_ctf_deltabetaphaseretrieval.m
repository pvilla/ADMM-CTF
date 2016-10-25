function x = operator_ctf_deltabetaphaseretrieval( b,z,lambda,pxs,beta_over_delta,FPSF )

    if nargin==5
        FPSF=[];
    end
    %%  Create map of wave numbers
    [ n1 , n2 ] = size( b );
    f1=ifftshift([-fix(n2/2):ceil(n2/2)-1])/n2;
    f2=ifftshift([-fix(n1/2):ceil(n1/2)-1])/n1;
    [f1,f2]=meshgrid(f1,f2);
    fmap = f1.^2 + f2.^2;
    clear f1 f2; 
    sinfmap=sin(pi*lambda*z*fmap/pxs^2);
    cosfmap=cos(pi*lambda*z*fmap/pxs^2);
    numerator=2*(sinfmap-beta_over_delta*cosfmap);
    if ~isempty(FPSF)
        %numerator=numerator.*fft2(ifftshift(FPSF));
        numerator=numerator.*FPSF;
    end
    %clear fmap;
    %%  FFT method
    bf = fft2( ifftshift(b) );
    xf = bf.*numerator;
    x  = real( fftshift(ifft2( xf ) ));
end


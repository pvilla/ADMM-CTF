function x = ctf_fixedratio_retrieval( b,alpha,z,lambda,pxs,betaoverdelta,FPSF )
    if nargin==6
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
    denominator=2*(sinfmap-betaoverdelta*cosfmap);
    if ~isempty(FPSF)
        denominator=denominator.*FPSF;
    end
    denominator(abs(denominator)<alpha & denominator>=0)=alpha;
    denominator(abs(denominator)<alpha & denominator<0)=-alpha;
    %clear fmap;
    %%  FFT method
    bf = fft2( ifftshift(b) );
    %bf(1) = 0;
    xf = bf./denominator;
    x  = real( fftshift(ifft2( xf ) ));
    x  = x - min(min(x));% shift the minimum value to zero, due to
                         % lose of quantitativeness (not well
                         % retrieved zero frequency
end


function kg = kernel_grad()
%kgg = kernel_gauss;
%kg = [ 1 0 1 ; 0 0 0 ; -1 0 -1 ];
    kg = [ 1 0 -1 ; 2 0 -2 ; 1 0 -1 ];
    kg = kg';
%    kg = conv2( kg , kgg );
end

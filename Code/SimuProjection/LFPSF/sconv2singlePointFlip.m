function C = sconv2singlePointFlip(sizeA, point, B, flipBX, flipBY, shape)
% Adapted from C = sconv2(A, B, shape) Author: Bruno Luong <brunoluong@yahoo.com>

m = sizeA(1);
n = sizeA(2);

[p, q] = size(B);

i = point(1);
j = point(2);
a = 1;
[k, l, b] = find(B);

if flipBX ~= 0
    k = p - k + 1;
end
if flipBY ~= 0
    l = q - l + 1;
end

[I, K] = ndgrid(i, k);
[J, L] = ndgrid(j, l);
C = a(:)*b(:).';

switch lower(shape)
    case 'full'
        C = sparse(I(:)+K(:)-1,J(:)+L(:)-1, C(:), m+p-1, n+q-1);
    case 'valid'
        mnc = max([m-max(0,p-1),n-max(0,q-1)],0);
        i = I(:)+K(:)-p;
        j = J(:)+L(:)-q;
        b = i > 0 & i <= mnc(1) & ...
            j > 0 & j <= mnc(2);
        C = sparse(i(b), j(b), C(b), mnc(1), mnc(2));
     case 'same'
        i = I(:)+K(:)-ceil((p+1)/2);
        j = J(:)+L(:)-ceil((q+1)/2);
        b = i > 0 & i <= m & ...
            j > 0 & j <= n;
        C = sparse(round(i(b)), round(j(b)), C(b), m, n);       
end
end
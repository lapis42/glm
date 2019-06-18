clc; clearvars; close all;

t1 = [0];
nt1 = numel(t1);

t2 = [0.05, 0.15, 0.45];
nt2 = numel(t2);

binsize = 0.1;
w = 0.4;
nbins = 2*floor(w / binsize) + 1;


m = - binsize * floor(nbins / 2);
B = zeros(nbins, 1);
for j = 1:nbins
    B(j) = m + (j - 1) * binsize;
end

i2 = 1;
C = zeros(nbins, nt1);
for i1 = 1:nt1
    lbound = t1(i1) - w - binsize/2;
    while (i2 < nt2 && t2(i2) < lbound)
        i2 = i2 + 1;
    end
    while (i2 > 1 && t2(i2-1) > lbound)
        i2 = i2 - 1;
    end
    
    rbound = lbound;
    l = i2;
    
    for j = 1:nbins
        k = 0;
        rbound = rbound + binsize;
        while(l <= nt2 && t2(l) < rbound)
            l = l + 1;
            k = k + 1;
        end
        
        C(j, i1) = k / binsize;
    end
end
        
plot(B, C);
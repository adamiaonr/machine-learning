% template for the discriminant function between class a vs. class b.
% returns 1 if x is classified as belonging to class 'a', 0 if 'b'.

% classes a and b which are discriminated by dscrmnt() are determined by 
% setting the appropriate input parameters Uk, iCovk and C.
% e.g. the discriminant function between classes 1 and 2 is 
% dscrmnt(x,U1,U2,iCov1,iCov2,C12).
function [dscrmnt] = dscrmnt(x,Ua,Ub,iCova,iCovb,C)

if (-0.5*((x-Ua)*iCova*(x-Ua)' - (x-Ub)*iCovb*(x-Ub)') - C) >= 0
    
    dscrmnt = 1;

else
    
    dscrmnt = 0;

end
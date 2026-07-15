function idx = centerBlockIndices(Ndim, nCenter)
%CENTERBLOCKINDICES Return 1-based contiguous center indices.
% For even nCenter, this matches the common ACS convention:
% centerIdx-n/2 : centerIdx+n/2-1 with centerIdx=floor(N/2)+1.
    centerIdx = floor(Ndim/2) + 1;
    if mod(nCenter, 2) == 0
        idx = (centerIdx - nCenter/2) : (centerIdx + nCenter/2 - 1);
    else
        idx = (centerIdx - floor(nCenter/2)) : (centerIdx + floor(nCenter/2));
    end
    assert(idx(1) >= 1 && idx(end) <= Ndim, ...
        'Requested center block exceeds dimension.');
end

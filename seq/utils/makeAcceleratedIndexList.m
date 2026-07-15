function idx = makeAcceleratedIndexList(N, R)
    if R < 1 || R ~= round(R)
        error('Acceleration factor must be a positive integer.');
    end
    centerIdx = floor(N/2) + 1;
    idx = [];
    for ii = 1:N
        if mod(ii - centerIdx, R) == 0
            idx(end+1) = ii; %#ok<AGROW>
        end
    end
end

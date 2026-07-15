function idx = makeCenterIndexList(N, Ncenter)
    if Ncenter < 1 || Ncenter ~= round(Ncenter)
        error('Center block size must be a positive integer.');
    end
    Ncenter = min(Ncenter, N);
    startIdx = floor(N/2) + 1 - floor(Ncenter/2);
    startIdx = max(1, min(startIdx, N - Ncenter + 1));
    idx = startIdx:(startIdx + Ncenter - 1);
end

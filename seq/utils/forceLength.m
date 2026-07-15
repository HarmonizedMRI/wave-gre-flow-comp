function wave = forceLength(wave, nTarget)
    wave = wave(:).';
    if numel(wave) > nTarget
        wave = wave(1:nTarget);
    elseif numel(wave) < nTarget
        wave = [wave, zeros(1, nTarget - numel(wave))];
    end
end

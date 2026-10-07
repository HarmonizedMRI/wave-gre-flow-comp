clear; clc

addpath('/Users/yiyund/Code_mgh/pulseq/matlab');
evaluationPath = fileparts(mfilename('fullpath'));
repoPath = fileparts(evaluationPath);
addpath(fullfile(repoPath, 'seq', 'utils'));

grePath = fullfile(evaluationPath, 'output', 'v1.5.1', ...
    'high_slew_wave_gre_sag');
referenceRoot = '/Users/yiyund/Code_mgh/sources/seq_for_eva';
calibrationPath = fullfile(referenceRoot, 'outputs', 'flash_calibration', ...
    'v1.5.1', 'high_slew');
cases = struct( ...
    'cycles', {10, 20, 25}, ...
    'amplitude', {12.732, 6.3662, 5.093}, ...
    'tag', {'12p732', '6p3662', '5p093'});
states = {'sinzero', 'sinctr'};

dt = 10e-6;
gammaHz = 42.576e6;
readoutDuration = 5e-3;
physicalSlew = 180;
lowPnsSlew = 63;
nImage = 32000;
nCalibration = 1312;
labelNames = {'LIN','PAR','ECO','AVG','SET','REF','IMA'};

for caseIndex = 1:numel(cases)
    waveCase = cases(caseIndex);
    stateResults = struct([]);

    for stateIndex = 1:numel(states)
        state = states{stateIndex};
        centered = strcmp(state, 'sinctr');
        greName = sprintf([ ...
            'gre_waveFC_SAG_FOV200x240x250_E2_R3x1_os4_', ...
            'A%s_C%d_%s_skyra_v151.seq'], ...
            waveCase.tag, waveCase.cycles, state);
        calibrationName = sprintf([ ...
            'gre3d_flashcal_SAG_FOV200x240x250_N250_os4_', ...
            'A%s_C%d_%s_skyra_v151.seq'], ...
            waveCase.tag, waveCase.cycles, state);
        assert(strlength(string(greName)) <= 100, ...
            'Wave-GRE filename exceeds 100 characters.');
        greFile = fullfile(grePath, greName);
        calibrationFile = fullfile(calibrationPath, calibrationName);
        assert(exist(greFile, 'file') == 2, ...
            'Missing Wave-GRE sequence: %s', greFile);
        assert(exist(calibrationFile, 'file') == 2, ...
            'Missing standalone calibration: %s', calibrationFile);

        sequence = mr.Sequence();
        sequence.read(greFile);
        calibration = mr.Sequence();
        calibration.read(calibrationFile);
        [timingOk, timingErrors] = sequence.checkTiming;
        assert(timingOk, 'Reloaded timing failed for %s:\n%s', ...
            greFile, strjoin(timingErrors, newline));
        [calTimingOk, calTimingErrors] = calibration.checkTiming;
        assert(calTimingOk, ...
            'Reloaded calibration timing failed for %s:\n%s', ...
            calibrationFile, strjoin(calTimingErrors, newline));

        fov = sequence.getDefinition('FOV');
        assert(max(abs(fov(:).'-1e-3*[200 240 250])) < 1e-12 && ...
            isequal([sequence.getDefinition('Nx'), ...
            sequence.getDefinition('Ny'), sequence.getDefinition('Nz')], ...
            [200 240 250]), 'SAG FOV/matrix definitions changed.');
        assert(strcmp(sequence.getDefinition('ReadoutAxis'), 'z') && ...
            strcmp(sequence.getDefinition('InnerPEAxis'), 'x') && ...
            strcmp(sequence.getDefinition('OuterPEAxis'), 'y'), ...
            'SAG axis definitions changed.');
        assert(sequence.getDefinition('WaveCycles') == waveCase.cycles && ...
            abs(sequence.getDefinition('WaveAmplitude_mTm') ...
            - waveCase.amplitude) < 1e-12, ...
            'Wave case definition mismatch.');
        assert(sequence.getDefinition('WaveCenteredOnNowave') == centered && ...
            sequence.getDefinition('CalibrationWaveCenteredOnNowave') ...
            == centered, 'Image/calibration centering states disagree.');
        assert(sequence.getDefinition('WaveCosCenterPolarity') == ...
            (-1)^waveCase.cycles, 'Cosine parity definition mismatch.');
        assert(sequence.getDefinition('UseFullInitialFC') == 1 && ...
            sequence.getDefinition('UseFullInterEchoFC') == 1, ...
            'The high-slew Wave-GRE must keep all FC modules enabled.');
        assert(sequence.getDefinition('ActiveWaveSlewLimit_Tms') == ...
            physicalSlew && ...
            sequence.getDefinition('WaveRampSlewLimit_Tms') == ...
            lowPnsSlew && ...
            sequence.getDefinition('NonWaveSlewLimit_Tms') == lowPnsSlew, ...
            'High-slew/low-PNS system definitions mismatch.');
        storedTE = sequence.getDefinition('TE');
        assert(max(abs(storedTE(:).'-[10 20]*1e-3)) ...
            < 1e-12 && abs(sequence.getDefinition('TR')-30e-3) < 1e-12, ...
            'Wave-GRE TE/TR changed.');

        labels = sequence.evalLabels('evolution', 'adc');
        calibrationLabels = calibration.evalLabels('evolution', 'adc');
        assert(numel(labels.LIN) == nImage+nCalibration, ...
            'Combined ADC count changed.');
        imageRange = 1:nImage;
        calibrationRange = nImage+(1:nCalibration);
        assert(all(labels.REF(imageRange) == 0) && ...
            all(labels.IMA(imageRange) == 0) && ...
            all(labels.SET(imageRange) == 0) && ...
            all(labels.AVG(imageRange) == 0), ...
            'Wave-GRE image routing changed.');
        assert(numel(unique(labels.LIN(imageRange))) == 80 && ...
            numel(unique(labels.PAR(imageRange))) == 200 && ...
            isequal(unique(labels.ECO(imageRange)), [0 1]), ...
            'Wave-GRE sampled LIN/PAR/ECO extents changed.');
        for echo = 0:1
            echoMask = labels.ECO(imageRange) == echo;
            pairs = [labels.PAR(imageRange(echoMask)).', ...
                labels.LIN(imageRange(echoMask)).'];
            assert(size(unique(pairs, 'rows'), 1) == 16000, ...
                'Echo %d has duplicate or missing PE pairs.', echo);
        end
        for labelIndex = 1:numel(labelNames)
            labelName = labelNames{labelIndex};
            integrated = labels.(labelName)(calibrationRange);
            assert(isequal(integrated(:), ...
                calibrationLabels.(labelName)(:)), ...
                'Calibration label mismatch for %s.', labelName);
        end

        centerOrdinals = zeros(1, 2);
        centerBlocks = cell(1, 2);
        centerBlockIndices = zeros(1, 2);
        centerM0 = zeros(1, 2);
        centerM1 = zeros(1, 2);
        for echo = 0:1
            centerOrdinals(echo+1) = find( ...
                labels.LIN(imageRange) == 120 & ...
                labels.PAR(imageRange) == 100 & ...
                labels.ECO(imageRange) == echo, 1);
            assert(centerOrdinals(echo+1) > 0, ...
                'Image mask is missing the center line for echo %d.', echo);
            [centerBlocks{echo+1}, centerBlockIndices(echo+1)] = ...
                findAdcBlock(sequence, centerOrdinals(echo+1));
            [centerM0(echo+1), centerM1(echo+1)] = ...
                yMomentsAfterRfToAdcCenter( ...
                sequence, centerBlockIndices(echo+1));
        end
        expectedCenterM0 = sequence.getDefinition( ...
            'WaveSinCenterTarget_1pm');
        assert(max(abs(centerM0-expectedCenterM0)) < 1e-4, ...
            ['Serialized image sine center M0 %s differs from target ', ...
            '%.12g 1/m.'], mat2str(centerM0, 12), expectedCenterM0);
        assert(max(abs(centerM1)) < 5e-7, ...
            ['Center-line y M1 is not flow compensated at both echoes: ', ...
            '%s 1/m*s.'], mat2str(centerM1, 12));

        imageBlock = centerBlocks{1};
        nPre = round(imageBlock.adc.delay/dt);
        nActive = round(readoutDuration/dt)+1;
        sineActive = imageBlock.gy.waveform(nPre+(1:nActive));
        cosineActive = imageBlock.gx.waveform(nPre+(1:nActive));
        sinePreSlew = arbitrarySlewPeak( ...
            imageBlock.gy.waveform(1:nPre), 0, 0, dt)/gammaHz;
        cosinePreSlew = arbitrarySlewPeak( ...
            imageBlock.gx.waveform(1:nPre), 0, cosineActive(1), dt) ...
            /gammaHz;
        sineActiveSlew = arbitrarySlewPeak( ...
            sineActive, 0, 0, dt)/gammaHz;
        cosineActiveSlew = arbitrarySlewPeak( ...
            cosineActive, cosineActive(1), cosineActive(end), dt) ...
            /gammaHz;
        cosinePost = imageBlock.gx.waveform(nPre+nActive+1:end);
        cosinePostSlew = arbitrarySlewPeak( ...
            cosinePost, cosineActive(end), 0, dt)/gammaHz;
        assert(max([sinePreSlew cosinePreSlew cosinePostSlew]) ...
            <= lowPnsSlew*1.001, ...
            'An image wave ramp exceeds the low-PNS slew limit.');
        assert(max([sineActiveSlew cosineActiveSlew]) <= ...
            physicalSlew*1.001 && ...
            min([sineActiveSlew cosineActiveSlew]) > 150, ...
            'Active-wave slew is outside the intended envelope.');

        assert(abs(sequence.getDefinition('CalibrationTE') ...
            - calibration.getDefinition('CalibrationTE')) < 1e-12 && ...
            abs(sequence.getDefinition('CalibrationTR') ...
            - calibration.getDefinition('CalibrationTR')) < 1e-12, ...
            'Integrated and standalone calibration timing differs.');
        [duration, nBlocks] = sequence.duration();
        [~, calibrationBlocks] = calibration.duration();
        assert(calibrationBlocks == 6648, ...
            'Unexpected standalone calibration block count.');
        for offset = 0:calibrationBlocks-1
            integratedBlock = normalizeDecodedBlock( ...
                sequence.getBlock(nBlocks-offset));
            standaloneBlock = normalizeDecodedBlock( ...
                calibration.getBlock(calibrationBlocks-offset));
            assert(decodedBlocksPhysicallyMatch( ...
                integratedBlock, standaloneBlock), ...
                ['Integrated calibration physical block mismatch at ', ...
                'reverse offset %d.'], offset);
        end

        stateResults(stateIndex).duration = duration;
        stateResults(stateIndex).blocks = nBlocks;
        stateResults(stateIndex).labels = labels;
        stateResults(stateIndex).sineActive = sineActive;
        stateResults(stateIndex).cosineWave = imageBlock.gx.waveform;

        fprintf([ ...
            'C%d %-7s passed: Gy M0=%s, M1=%s; active sine/cos ', ...
            'slew=%.6f/%.6f T/m/s; %.6f s, %d blocks; ', ...
            'cal tail=%d blocks matched.\n'], ...
            waveCase.cycles, state, mat2str(centerM0, 9), ...
            mat2str(centerM1, 4), sineActiveSlew, cosineActiveSlew, ...
            duration, nBlocks, calibrationBlocks);
    end

    assert(abs(stateResults(1).duration-stateResults(2).duration) < 1e-9 && ...
        stateResults(1).blocks == stateResults(2).blocks, ...
        'Sine centering changed total duration or block count.');
    for labelIndex = 1:numel(labelNames)
        labelName = labelNames{labelIndex};
        assert(isequal(stateResults(1).labels.(labelName), ...
            stateResults(2).labels.(labelName)), ...
            'Sine centering changed combined %s labels.', labelName);
    end
    assert(max(abs(stateResults(1).sineActive ...
        - stateResults(2).sineActive)) < 1e-8, ...
        'Sine centering changed active sine samples.');
    assert(max(abs(stateResults(1).cosineWave ...
        - stateResults(2).cosineWave)) < 1e-8, ...
        'Sine centering changed the cosine waveform.');
end

fprintf(['High-slew full-FC Wave-GRE validation passed for all six ', ...
    'files. PNS and forbidden-frequency checks were not run.\n']);

function [targetBlock, targetBlockIndex] = findAdcBlock(sequence, targetAdc)
    [~, nBlocks] = sequence.duration();
    adcCounter = 0;
    for blockIndex = 1:nBlocks
        block = sequence.getBlock(blockIndex);
        if ~isempty(block.adc)
            adcCounter = adcCounter+1;
            if adcCounter == targetAdc
                targetBlock = block;
                targetBlockIndex = blockIndex;
                return;
            end
        end
    end
    error('Requested ADC block was not decoded.');
end

function [M0, M1] = yMomentsAfterRfToAdcCenter(sequence, adcBlockIndex)
    rfBlockIndex = adcBlockIndex-1;
    while rfBlockIndex >= 1 && ...
            isempty(sequence.getBlock(rfBlockIndex).rf)
        rfBlockIndex = rfBlockIndex-1;
    end
    assert(rfBlockIndex >= 1, ...
        'Could not locate the RF block preceding the selected ADC.');

    blockStart = 0;
    tEcho = 0;
    eventTimes = {};
    eventAmplitudes = {};
    for blockIndex = rfBlockIndex+1:adcBlockIndex
        block = sequence.getBlock(blockIndex);
        if blockIndex == adcBlockIndex
            localEnd = block.adc.delay ...
                +0.5*block.adc.numSamples*block.adc.dwell;
            tEcho = blockStart+localEnd;
        else
            localEnd = mr.calcDuration(block);
        end
        if ~isempty(block.gy)
            [times, amplitudes] = gradientPolyline(block.gy);
            keep = times <= localEnd+1e-12;
            times = times(keep);
            amplitudes = amplitudes(keep);
            if times(end) < localEnd-1e-12
                amplitudes(end+1) = interp1( ... %#ok<AGROW>
                    gradientPolylineTimes(block.gy), ...
                    gradientPolylineAmplitudes(block.gy), ...
                    localEnd, 'linear', 0);
                times(end+1) = localEnd; %#ok<AGROW>
            end
            eventTimes{end+1} = blockStart+times; %#ok<AGROW>
            eventAmplitudes{end+1} = amplitudes; %#ok<AGROW>
        end
        blockStart = blockStart+mr.calcDuration(block);
    end

    M0 = 0;
    M1 = 0;
    for eventIndex = 1:numel(eventTimes)
        [eventM0, eventM1] = continuousMomentFromPolylineWindow( ...
            eventTimes{eventIndex}, eventAmplitudes{eventIndex}, ...
            0, tEcho, tEcho);
        M0 = M0+eventM0;
        M1 = M1+eventM1;
    end
end

function [times, amplitudes] = gradientPolyline(gradient)
    times = gradientPolylineTimes(gradient);
    amplitudes = gradientPolylineAmplitudes(gradient);
end

function times = gradientPolylineTimes(gradient)
    if strcmp(gradient.type, 'trap')
        times = gradient.delay+[0, gradient.riseTime, ...
            gradient.riseTime+gradient.flatTime, ...
            gradient.riseTime+gradient.flatTime+gradient.fallTime];
    else
        times = gradient.delay+[0; gradient.tt(:); gradient.shape_dur].';
    end
    [times, uniqueIndex] = unique(times, 'stable');
    amplitudes = gradientPolylineAmplitudesRaw(gradient);
    amplitudes = amplitudes(uniqueIndex); %#ok<NASGU>
end

function amplitudes = gradientPolylineAmplitudes(gradient)
    timesRaw = gradientPolylineTimesRaw(gradient);
    amplitudes = gradientPolylineAmplitudesRaw(gradient);
    [~, uniqueIndex] = unique(timesRaw, 'stable');
    amplitudes = amplitudes(uniqueIndex);
end

function times = gradientPolylineTimesRaw(gradient)
    if strcmp(gradient.type, 'trap')
        times = gradient.delay+[0, gradient.riseTime, ...
            gradient.riseTime+gradient.flatTime, ...
            gradient.riseTime+gradient.flatTime+gradient.fallTime];
    else
        times = gradient.delay+[0; gradient.tt(:); gradient.shape_dur].';
    end
end

function amplitudes = gradientPolylineAmplitudesRaw(gradient)
    if strcmp(gradient.type, 'trap')
        amplitudes = [0, gradient.amplitude, gradient.amplitude, 0];
    else
        amplitudes = [gradient.first; gradient.waveform(:); ...
            gradient.last].';
    end
end

function slewPeak = arbitrarySlewPeak(waveform, first, last, dt)
    waveform = waveform(:);
    if isempty(waveform)
        slewPeak = abs(last-first)/dt;
        return;
    end
    slew = [2*(first-waveform(1)); ...
        waveform(2:end)-waveform(1:end-1); ...
        2*(waveform(end)-last)]/dt;
    slewPeak = max(abs(slew));
end

function block = normalizeDecodedBlock(block)
    if isfield(block, 'label')
        block = rmfield(block, 'label');
    end
    gradientFields = {'gx','gy','gz'};
    for ii = 1:numel(gradientFields)
        name = gradientFields{ii};
        if isstruct(block.(name))
            metadataFields = intersect(fieldnames(block.(name)), ...
                {'id','shape_id','time_id'});
            if ~isempty(metadataFields)
                block.(name) = rmfield(block.(name), metadataFields);
            end
            % Integer-cycle sine integration can leave a signed numerical
            % zero in the post trap. Normalize only sub-micro-Hz/m events;
            % all nonzero physical gradients remain exact comparisons.
            if strcmp(block.(name).type, 'trap') && ...
                    abs(block.(name).amplitude) < 1e-6
                block.(name).amplitude = 0;
                block.(name).area = 0;
                block.(name).flatArea = 0;
            end
        end
    end
end

function matched = decodedBlocksPhysicallyMatch(blockA, blockB)
    if isequaln(blockA, blockB)
        matched = true;
        return;
    end

    gradientFields = {'gx','gy','gz'};
    amplitudeFields = {'amplitude','waveform','first','last'};
    areaFields = {'area','flatArea'};
    timeFields = {'riseTime','flatTime','fallTime','delay','shape_dur','tt'};
    for gradientIndex = 1:numel(gradientFields)
        name = gradientFields{gradientIndex};
        gradA = blockA.(name);
        gradB = blockB.(name);
        if isempty(gradA) || isempty(gradB)
            if ~isequaln(gradA, gradB)
                matched = false;
                return;
            end
            continue;
        end
        if ~strcmp(gradA.type, gradB.type) || ...
                ~strcmp(gradA.channel, gradB.channel)
            matched = false;
            return;
        end
        for fieldIndex = 1:numel(amplitudeFields)
            fieldName = amplitudeFields{fieldIndex};
            if isfield(gradA, fieldName) ~= isfield(gradB, fieldName) || ...
                    (isfield(gradA, fieldName) && ...
                    (~isequal(size(gradA.(fieldName)), ...
                    size(gradB.(fieldName))) || ...
                    max(abs(gradA.(fieldName)(:) ...
                    - gradB.(fieldName)(:))) > 1e-6))
                matched = false;
                return;
            end
        end
        for fieldIndex = 1:numel(areaFields)
            fieldName = areaFields{fieldIndex};
            if isfield(gradA, fieldName) ~= isfield(gradB, fieldName) || ...
                    (isfield(gradA, fieldName) && ...
                    abs(gradA.(fieldName)-gradB.(fieldName)) > 1e-8)
                matched = false;
                return;
            end
        end
        for fieldIndex = 1:numel(timeFields)
            fieldName = timeFields{fieldIndex};
            if isfield(gradA, fieldName) ~= isfield(gradB, fieldName) || ...
                    (isfield(gradA, fieldName) && ...
                    (~isequal(size(gradA.(fieldName)), ...
                    size(gradB.(fieldName))) || ...
                    max(abs(gradA.(fieldName)(:) ...
                    - gradB.(fieldName)(:))) > 1e-12))
                matched = false;
                return;
            end
        end
        blockB.(name) = blockA.(name);
    end
    matched = isequaln(blockA, blockB);
end

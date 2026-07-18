% gre_3d_wave_with_flash_calibration.m
% Author: Yiyun Dong
% Affiliation: Athinoula A. Martinos Center for Biomedical Imaging
% Date: 2026-07-15
%
% 3D multi-echo wave-encoded GRE with full flow compensation and an
% integrated slab-selective FLASH wave-calibration acquisition.
%
% GRE implementation built based on Berkin's GRE code:
% https://github.com/HarmonizedMRI/megre_label/blob/main/script_writeGradientEcho3D_label_spoil_github_v0.m
%
% Integration structure and calibration helper organization follow:
% https://github.com/HarmonizedMRI/wave-mprage
%
% Acquisition order:
%   1. GRE dummy scans and image acquisition (TWIX image)
%   2. FLASH calibration dummy/settling scans and five calibration SETs
%      (TWIX refscan)
%
% TWIX routing and expected calibration layout:
%   GRE image: REF=false, IMA=false, SET=0, ECO=0:(Nechoes-1)
%   Calibration: REF=true, IMA=false, ECO=0, SET=0:4
%     SET 0: no-wave, LIN-wide / PAR-narrow
%     SET 1: sine-wave, LIN-wide / PAR-narrow
%     SET 2: no-wave, PAR-wide / LIN-narrow
%     SET 3: cosine-wave, PAR-wide / LIN-narrow
%     SET 4: no-wave ACS
%   With Ncalib1=72 and NacsCal=32, the logical refscan extent is
%   [LIN=72, PAR=72, SET=5]. A typical loader shape is
%   [Nx_os, Ncoil, 72, 72, 5]; unacquired stripe entries remain zero.
%
% Geometry is TRA for both acquisitions:
%   RO=x, LIN=y/sine, PAR=z/cosine/slab-select.
%   GRE and calibration LIN/PAR indices both map from negative to positive
%   physical k-space as the MATLAB/TWIX array indices increase.
%
% The calibration slab rephaser is a standalone block and is not overlapped
% with the following calibration PE/readout/wave gradients.
%
% Local helper functions are stored in ./utils/.
%
% Do not call clear/clear all here: users may predefine path variables in the
% MATLAB workspace before running this script.
close all; clc
format long

%% Path
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
utils_path = fullfile(script_dir, 'utils');
if ~exist(utils_path, 'dir')
    error('Required local utils folder not found: %s', utils_path);
end
addpath(utils_path);

% Saved beside this script as gre_flash_path_settings.json.
pathSettings = configurePathSettings(script_dir);
pulseq_path = pathSettings.pulseq_path;
safe_pns_prediction_path = pathSettings.safe_pns_prediction_path;
out_path = ensureTrailingFilesep(pathSettings.out_path);
system_asc_file = pathSettings.system_asc_file;

pulseq_matlab_path = fullfile(pulseq_path, 'matlab');
if exist(fullfile(pulseq_matlab_path, '+mr'), 'dir')
    addpath(pulseq_matlab_path);
elseif exist(fullfile(pulseq_path, '+mr'), 'dir')
    addpath(pulseq_path);
else
    warning(['Could not find +mr under the provided Pulseq path. ', ...
        'Continuing after adding the provided path.']);
    addpath(pulseq_path);
end
if ~isempty(safe_pns_prediction_path)
    addpath(safe_pns_prediction_path);
end

%% Parameters

% Write options:
%   write_v141_format = false  -> write only current format (v1.5.x)
%   write_v141_format = true   -> also write legacy v1.4.1 format
write_v141_format = true;

% tag_calib = 'test_';
% tag_calib = 'FCWaveRead_';
tag_calib = '';

% ------------------------- sequence timing -------------------------
alpha          = 15;                         % [deg]
% TE             = [10 20] * 1e-3;       % [s]
TE             = [20] * 1e-3;       % [s]
Nechoes        = numel(TE);
TR             = 30e-3;                      % [s]
rfSpoilingInc  = 50;                         % [deg]
NdummyGre      = 300;
naverage       = 1;

% High-TBP slab-selective excitation.
rfDuration     = 1.0e-3;                     % [s]
rfTBP          = 20;
rfApodization  = 0.42;

% Monopolar readout. roDuration is the full oversampled ADC duration:
%   256 readout points * 4x RO oversampling * 5 us dwell = 5.12 ms.
% Echo spacing must exceed roDuration plus the x prephaser/flyback/ramp terms.
% roDuration     = 5120e-6;                    % [s] (5 us dwell, 195 Hz, os=4, Nx=256)
roDuration     = 5000e-6;                    % [s] (5 us dwell, 200 Hz, os=4, Nx=250)
% roDuration     = 3840e-6;                    % [s] (5 us dwell, 195 Hz, os=3, Nx=256)
% roDuration     = 3240e-6;                    % [s] (4.5 us dwell, 333 Hz, os=3, Nx=240)
os_factor      = 4;
gre_ro_spoil   = 2.0;                        % extra readout-spoiler kmax units

% Reference duration for fixed-duration PE/spoiler timing. The split x/cos
% prep module below is minimum-time and no longer forced to TxPre.
TxPre          = 1.91e-3;  %1.69e-3 for prisma/cimax, 1.91e-3 for skyra;
TPre           = TxPre;

% Optional gap between the prep/dephaser module and the readout/wave module.
% Used for echo 1 and for all echoes when inter-echo FC is disabled. When
% centered inter-echo FC is enabled, echoes 2+ ignore this user gap and instead
% center the shared x-flyback/cosine-prewinder/sine-FC module between neighboring
% readout gradients.
prepToReadoutGap = 0;

% ------------------------- flow compensation -------------------------
% Step 1: for echo 2+, center the common-duration x flyback and cosine
% prewinder in the gap between neighboring readout gradients.
% Step 3: optionally add a zero-area y/sine FC lobe in the same centered
% inter-echo module to cancel the sine readout-wave M1.
% Step 4: optionally add a zero-area y/sine FC lobe before echo 1.
% Step 6: optionally redesign the echo-1 x prephaser and z/cosine prewinder
% by preserving their M0 and overlapping a zero-M0 bipolar M1 correction.
% Step 9b/9c: optionally move y/LIN and z/PAR initial prephasers into
% the right-edge-aligned initial prep module and merge them with sine/cos FC.
% Step 9d: optionally replace the ordinary slab rephaser with an M0/M1
% slab-select FC rephaser glued immediately after gz_ss.
% Step 10: optionally add PE-y drift compensation into the inter-echo
% zero-M0 y FC module. One max-|M1| waveform is designed per inter-echo gap
% and scaled over PE-y indices so timing remains echo-specific and fixed.
% Step 11: optionally add PAR-z drift compensation into the inter-echo
% z/cosine module. Because the cosine module has nonzero M0, one waveform
% is designed per echo gap and per PAR-z index with the same timing.
isUseFlowComp = true;
isFCInterEchoReadoutCos = true;
isFCInterEchoSine = true;
isFCInterEchoPEY = true;        % Step 10: add PE-y M0*ESP term into inter-echo y FC
isFCInterEchoPARZ = true;       % Step 11: add PAR-z M0*ESP term into inter-echo z/cos FC
isApplyInterEchoCosEndpointM1Correction = true; % Step 11d: compensate the half-raster G0 endpoint bias in inter-echo cosine M1
isFCInitialSine = true;
isFCInitialReadout = true;
isFCInitialCosine = true;
isFCInitialPEY = true;          % Step 9 part 1: PE-y M1 FC in the initial echo prep module
isFCInitialPARZ = true;         % Step 9 part 2a: merge z/PAR prephaser with initial cosine FC prephaser
isFCSlabRephZ = true;           % Step 9d: FC the late half of gz_ss with a glued slab rephaser

% PE/slab prewinder timing mode before echo 1.
%   'shortest' : use the shortest common feasible duration for gy/gz PE + slab rephaser.
%                This maximizes TE margin and is the default for TE1 = 6 ms.
%   'fixed'    : force gy/gz PE + slab rephaser to fixedPEPreDuration, typically TPre,
%                which can be useful for acoustic sinc-null timing. If this makes TE1
%                infeasible, the timing assert below will report the required minimum TE.
gPEPreTimingMode   = 'shortest';   % choices: 'shortest' or 'fixed'
fixedPEPreDuration = TPre;         % used only when gPEPreTimingMode = 'fixed'

% ------------------------- geometry -------------------------
slOrientation = 'TRA';                       % Siemens/Pulseq transverse mapping
ax = struct;
ax.d1 = 'x';                                 % readout
ax.d2 = 'z';                                 % inner PE / PAR / partition
ax.d3 = 'y';                                 % outer PE / LIN / phase
ax.n1 = strfind('xyz', ax.d1);
ax.n2 = strfind('xyz', ax.d2);
ax.n3 = strfind('xyz', ax.d3);

% target_fov = [220 172 160] * 1e-3;           % [m]
target_fov = [220 220 160] * 1e-3;           % [m]
sliceOS    = 0.125;

% Encoded z slab = 160 mm * (1 + 12.5%) = 180 mm, 64*(1+12.5%) = 72.
% fov = [220 172 180] * 1e-3;                 % encoded FOV [m] (children)
fov = [220 220 180] * 1e-3;                 % encoded FOV [m] (adults)

% Requested voxel size [x y z], in mm. The matrix N is derived from fov and
% res, with each dimension rounded to the nearest even integer so that the
% k-space center index remains well defined for the PE ordering/TI logic.
% Examples:
%   For children: 
%     res = [0.86  0.9   2.5 ];  % N ~= [256 192 72]
%     res = [0.86  1.25  2.5 ];  % N ~= [256 138 72]
%     res = [0.86  1.5   2.5 ];  % N ~= [256 114 72]
%   For adults: 
%     res = [0.86  0.86  2.5 ];  % N ~= [256 256 72]
%     res = [0.86  1.25  2.5 ];  % N ~= [256 138 72]
%     res = [0.86  1.5   2.5 ];  % N ~= [256 114 72]
%     res = [0.88  0.88  2.5 ];  % N ~= [250 250 72]
%     res = [0.92  0.92  2.5 ];  % N ~= [240 240 72]
% res = [0.92 0.92 2.5];          % requested resolution [x y z], in mm
res = [0.88 0.88 2.5];          % requested resolution [x y z], in mm
N = 2 * round((fov(:).' * 1e3 ./ res) / 2);
actualRes = fov(:).' ./ N * 1e3; % actual achieved resolution [x y z], in mm
fprintf('Requested resolution [x y z] = [%.4g %.4g %.4g] mm. Derived N = [%d %d %d]. Actual resolution = [%.4g %.4g %.4g] mm.\n', ...
    res(1), res(2), res(3), N(1), N(2), N(3), actualRes(1), actualRes(2), actualRes(3));

Nx    = N(1);
Ny    = N(2);
Nz    = N(3);
Nx_os = Nx * os_factor;

slabExciteThickness = target_fov(3);         % RF excites target slab [m]
slabEncodeThickness = fov(3);                % PAR encodes oversampled slab [m]

% ------------------------- acceleration -------------------------
Ry = 3;                                      % acceleration along y/LIN
Rz = 1;                                      % acceleration along z/PAR

% ------------------------- FLASH calibration -------------------------
% Calibration is appended after the GRE image acquisition and routed to
% TWIX refscan using compact local LIN/PAR labels inside SET=0:4.
NdummyCal      = 300;
NsettlePerPart = 10;
Ncalib1        = 72;
Ncalib2        = 1;
NacsCal        = 32;
calib_ro_spoil = 3.0;
calibWaveInfoFlag = false;
calibWaveDebugFlag = false;

% ------------------------- wave -------------------------
gwave_max     = 8;                           % [mT/m] (8 for long Tread, 7 for 3000 ms Tread)
swave_max     = 200;                         % [T/m/s]
Ncycles       = 10;
isUseWave_sin = true;                        % y-channel sine wave
isUseWave_cos = true;                        % z-channel cosine wave
% isUseWave_sin = false;                        % y-channel sine wave
% isUseWave_cos = false;                        % z-channel cosine wave
waveSinChannel = 'y';
waveCosChannel = 'z';

% Set true for verbose helper-function reports.
waveInfoFlag = true;
waveDebugFlag = true;

assert(NdummyCal >= 0 && NdummyCal == round(NdummyCal), ...
    'NdummyCal must be a nonnegative integer.');
assert(NsettlePerPart >= 0 && NsettlePerPart == round(NsettlePerPart), ...
    'NsettlePerPart must be a nonnegative integer.');
assert(Ncalib1 > Ncalib2 && Ncalib2 > 0 && ...
    Ncalib1 == round(Ncalib1) && Ncalib2 == round(Ncalib2), ...
    'Require positive integer calibration sizes with Ncalib1 > Ncalib2.');
assert(NacsCal > 0 && NacsCal == round(NacsCal), ...
    'NacsCal must be a positive integer.');
assert(Ncalib1 <= Ny && Ncalib1 <= Nz, ...
    'Ncalib1 exceeds a GRE PE dimension.');
assert(Ncalib2 <= Ny && Ncalib2 <= Nz, ...
    'Ncalib2 exceeds a GRE PE dimension.');
assert(NacsCal <= Ncalib1, 'NacsCal must not exceed Ncalib1.');

if naverage < 1 || naverage ~= round(naverage)
    error('naverage must be a positive integer.');
end
if NdummyGre < 0 || NdummyGre ~= round(NdummyGre)
    error('NdummyGre must be a non-negative integer.');
end
if os_factor ~= round(os_factor) || os_factor < 1
    error('os_factor must be a positive integer.');
end
if Nx_os ~= 1024
    warning('Nx_os = %d, expected 1024 for Nx=256 and os_factor=4.', Nx_os);
end
if abs(slabEncodeThickness/slabExciteThickness - (1 + sliceOS)) > 1e-9
    warning('Encoded/excited slab ratio does not match sliceOS.');
end

%% System limits

sys_type_options = {'prisma', 'skyra', 'Connectome2', 'C2_simulate_prisma', ...
    'trio', 'prisma_XA30A', 'premier', 'CimaX', 'TerraX'};
sys_type = selectStringOption('sys_type', 'Select scanner/system name', ...
    sys_type_options, 'prisma');
slew_safety_magrin        = 0.7;
grad_safety_magrin        = 0.9;
lowPNS_slew_safety_margin = 0.41;
lowPNS_grad_safety_margin = grad_safety_magrin;
diff_slew_safety_margin   = 0.45;
diff_grad_safety_margin   = 0.97;

if strcmp(sys_type,'prisma') || strcmp(sys_type,'C2_simulate_prisma') || strcmp(sys_type,'prisma_XA30A')
    physical_slew_max = 200;
    physical_grad_max = 80;
    B0=2.89;
elseif strcmp(sys_type,'premier')
    physical_slew_max = 200;
    physical_grad_max = 70;
    B0=3;
elseif strcmp(sys_type,'Connectome2')
    physical_slew_max = 598.802;
    physical_grad_max = 500;
    B0=2.89;
elseif strcmp(sys_type,'skyra')
    physical_slew_max = 180;
    physical_grad_max = 43;
    B0=2.89;
elseif strcmp(sys_type,'trio')
    physical_slew_max = 170;
    physical_grad_max = 38;
    B0=2.89;
elseif strcmp(sys_type,'CimaX')
    physical_slew_max = 200;
    physical_grad_max = 200;
    B0=2.89;
elseif strcmp(sys_type,'TerraX')
    physical_slew_max = 250;
    physical_grad_max = 135;
    B0=2.89;
else
    error('Undefined sys_type: %s', sys_type)
end

isGEscanner = strcmp(sys_type,'premier');
if ~isGEscanner
    pislquant = 0;
end
if isGEscanner
    psd_rf_wait = 200e-6;
    rfDeadTime = 100e-6;
    rfRingdownTime = 60e-6 + psd_rf_wait;
    adcDeadTime = 20e-6;
    adcRasterTime = 2e-6;
    rfRasterTime = 2e-6;
    gradRasterTime = 4e-6;
    blockDurationRaster = 4e-6;
else
    rfDeadTime = 100e-6;
    rfRingdownTime = 100e-6;
    adcDeadTime = 20e-6;
    adcRasterTime = 100e-9;
    rfRasterTime = 1e-6;
    gradRasterTime = 10e-6;
    blockDurationRaster = 10e-6;
end

sys = mr.opts('MaxGrad',physical_grad_max*grad_safety_magrin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*slew_safety_magrin,'SlewUnit','T/m/s',...
    'rfDeadTime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);

sys_lowPNS = mr.opts('MaxGrad',physical_grad_max*lowPNS_grad_safety_margin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*lowPNS_slew_safety_margin,'SlewUnit','T/m/s',...
    'rfDeadtime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);

sys_lowPNS2 = mr.opts('MaxGrad',physical_grad_max*0.6,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*0.35,'SlewUnit','T/m/s',...
    'rfDeadtime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);  % For FC module

sys_diff = mr.opts('MaxGrad',physical_grad_max*diff_grad_safety_margin,'GradUnit','mT/m',...
    'MaxSlew',physical_slew_max*diff_slew_safety_margin,'SlewUnit','T/m/s',...
    'rfDeadtime', rfDeadTime, ...
    'rfRingdownTime', rfRingdownTime, ...
    'adcDeadTime', adcDeadTime,...
    'adcRasterTime', adcRasterTime,...
    'rfRasterTime', rfRasterTime,...
    'gradRasterTime', gradRasterTime,...
    'blockDurationRaster', blockDurationRaster,...
    'B0',B0);

lims = sys;
seq = mr.Sequence(sys);

%% Setup RF, readout, PE, and ADC

[rf, gz_ss, gz_ssReph] = mr.makeSincPulse(alpha*pi/180, sys_lowPNS, ...
    'Duration', rfDuration, ...
    'SliceThickness', slabExciteThickness, ...
    'apodization', rfApodization, ...
    'timeBwProduct', rfTBP, ...
    'use', 'excitation');

% K-space units are 1/m in Pulseq MATLAB.
deltak = 1 ./ fov;

% Readout: full oversampled ADC duration is roDuration.
dwell = round((roDuration / Nx_os) / sys.adcRasterTime) * sys.adcRasterTime;
Tread = dwell * Nx_os;
if abs(Tread - roDuration) > sys.adcRasterTime
    fprintf('Requested roDuration %.6f ms rasterized to Tread %.6f ms.\n', roDuration*1e3, Tread*1e3);
end
fprintf('Readout setup: Nx=%d, os_factor=%d, Nx_os=%d, dwell=%.3f us, Tread=%.6f ms\n', ...
    Nx, os_factor, Nx_os, dwell*1e6, Tread*1e3);

gro = mr.makeTrapezoid(ax.d1, 'FlatArea', Nx*deltak(ax.n1), ...
    'FlatTime', Tread, 'system', sys);
adc = mr.makeAdc(Nx_os, 'Duration', Tread, 'Delay', gro.riseTime, 'system', sys);

% Monopolar readout preparation and flyback are no longer forced to
% TxPre. In this split-block/min-prep version they are rebuilt later with
% the shortest common duration required by the x prep and cosine prewinder
% for each echo.
gxPrepAreaEcho1 = -gro.area/2;
gxPrepAreaRest  = -gro.area;
groPreNatural = mr.makeTrapezoid(ax.d1, 'Area', gxPrepAreaEcho1, ...
    'system', sys_lowPNS);
groFlyBackNatural = mr.makeTrapezoid(ax.d1, 'Area', gxPrepAreaRest, ...
    'system', sys_lowPNS);

% Rasterize user-selected prep/readout gap. This gap is used for echo 1 and
% for all echoes when inter-echo FC is disabled.
prepToReadoutGap = round(prepToReadoutGap / sys.gradRasterTime) * sys.gradRasterTime;

% The readout/wave module uses the cosine ramp-up time as its pre-ADC
% alignment time. If cosine is disabled, fall back to the native x-readout
% ramp-up time so nowave/sine-only behavior remains natural.
if isUseWave_cos
    [G0_cos_timing, ~, ~] = designWaveAmplitude(Tread, sys, Ncycles, ...
        gwave_max, swave_max, physical_slew_max, 'cosine timing preview', false);
    [cosRampUpPreview, nCosRampUp, T_cosRampUp, cosRampUpSlewPeak] = ...
        makeShortestEndpointRampWave(0, G0_cos_timing, sys_lowPNS);
    gCosRampUpPreview = mr.makeArbitraryGrad(waveCosChannel, cosRampUpPreview, ...
        'system', sys_lowPNS, 'first', 0, 'last', G0_cos_timing);
    cosRampUpAreaPreview = gCosRampUpPreview.area;
else
    nCosRampUp = round(gro.riseTime / sys.gradRasterTime);
    T_cosRampUp = nCosRampUp * sys.gradRasterTime;
    cosRampUpSlewPeak = 0;
    cosRampUpAreaPreview = 0;
end

% Delay the x readout trapezoid so the start of its flat top aligns with the
% start of the actual cosine/sine ADC-window waveform.
readoutGradientDelay = round((T_cosRampUp - gro.riseTime) / sys.gradRasterTime) * sys.gradRasterTime;
assert(readoutGradientDelay >= -sys.gradRasterTime/10, ...
    ['Cosine ramp-up time %.6f ms is shorter than readout ramp-up %.6f ms. ', ...
     'Use a longer cosine ramp or a common-ramp design.'], ...
    T_cosRampUp*1e3, gro.riseTime*1e3);
readoutGradientDelay = max(0, readoutGradientDelay);

gxRead = gro;
gxRead.delay = readoutGradientDelay;
adc.delay = T_cosRampUp;

% The cosine area-prewinder is now a separate prep trapezoid. Its final
% duration is not known until we compare its shortest feasible duration with
% the readout dephaser/flyback duration. Use NaN here to request a natural
% shortest-duration provisional prewinder from defineCosineReadoutWave().
T_cosPreTrap = NaN;
T_wavePrePad = T_cosRampUp;

readoutModuleDurX = mr.calcDuration(gxRead);
tol = sys.gradRasterTime/10;
assert(abs(adc.delay - T_cosRampUp) < tol);
assert(abs(gxRead.delay + gro.riseTime - adc.delay) < tol, ...
    'Readout flat top is not aligned with ADC/wave start.');

fprintf(['Split readout/wave module timing:\n', ...
         '  cosine ramp-up samples       = %d grad-raster points\n', ...
         '  cosine ramp-up duration      = %.6f us\n', ...
         '  readout native rise time      = %.6f us\n', ...
         '  readout gradient delay        = %.6f us\n', ...
         '  adc.delay / wave pre-padding  = %.6f us\n', ...
         '  prep-to-readout gap           = %.6f us\n'], ...
    nCosRampUp, T_cosRampUp*1e6, gro.riseTime*1e6, readoutGradientDelay*1e6, ...
    adc.delay*1e6, prepToReadoutGap*1e6);
if isUseWave_cos
    fprintf('  preview cosine ramp-up area = %.9g 1/m, peak slew = %.6f T/m/s equiv\n', ...
        cosRampUpAreaPreview, cosRampUpSlewPeak / sys.gamma);
end

% Step 5/6 diagnostic: M1 of the first half of the readout gradient.
% Use the start of the split readout/wave module as the reference time so
% the target is directly comparable to the right-edge-aligned initial prep lobe.
t_adc_center = adc.delay + (adc.numSamples / 2) * adc.dwell;
% Readout-gradient support, used by the inter-echo FC duration search to
% compute the actual module-to-readout-start gap for a candidate duration.
tReadGradStartForFC = gxRead.delay;
tReadGradEndForFC   = mr.calcDuration(gxRead);
[tGxRead, aGxRead] = trapezoidCorners(gxRead);
[readoutFirstHalfM0, readoutFirstHalfM1] = continuousMomentFromPolylineWindow( ...
    tGxRead, aGxRead, 0, t_adc_center, 0);
readoutInitialFCTargetM1 = -readoutFirstHalfM1;

fprintf(['\nStep 5/6 readout/x initial-FC M1 diagnostic:\n', ...
         '  reference time = readout/wave module start\n', ...
         '  echo center in readout module          = %.6f ms\n', ...
         '  first-half readout M0                  = %.9g 1/m\n', ...
         '  first-half readout M1                  = %.9g 1/m*s\n', ...
         '  initial readout-FC target M1 (-M1)     = %.9g 1/m*s\n'], ...
    t_adc_center*1e3, readoutFirstHalfM0, readoutFirstHalfM1, readoutInitialFCTargetM1);

% Spoiler after the final echo. The same polarity is used because this is
% explicitly monopolar multi-echo GRE.
gxSpoilArea = gre_ro_spoil * Nx * deltak(ax.n1);
gxSpoil = mr.makeTrapezoid(ax.d1, 'Area', gxSpoilArea, ...
    'Duration', TPre, 'system', sys_lowPNS);

% PE areas: y/LIN and z/PAR use encoded FOV/matrix.
phaseAreasY = ((0:Ny-1) - Ny/2) * deltak(2);
phaseAreasZ = ((0:Nz-1) - Nz/2) * deltak(3);

% Sinc rephaser is played together with z/PAR prewinder before echo 1.
% The duration can be either the shortest common feasible value or a fixed
% user-selected duration such as TPre for acoustic sinc-null timing.
%
% Step 9d: if slab-rephaser FC is active, replace the ordinary M0-only
% gz_ssReph with an M0/M1-designed lobe that is glued to the end of the
% RF/gz_ss block. The reference for the slab subsystem is the rephaser start,
% i.e. the end of the RF/gz_ss block. Since the combined late-half-gz_ss +
% slab-rephaser M0 is zero, this zero-M1 condition is reference-invariant.
isUseSlabRephFCPreview = isUseFlowComp && isFCSlabRephZ;
rfBlockDurForSlabFC = mr.calcDuration(rf, gz_ss);
tRfCenterInBlockForSlabFC = mr.calcRfCenter(rf) + rf.delay;
[tGzSsForSlabFC, aGzSsForSlabFC] = trapezoidCorners(gz_ss);
tSlabRephStartForFC = max(rfBlockDurForSlabFC, tGzSsForSlabFC(end));
if tGzSsForSlabFC(end) < tSlabRephStartForFC - eps
    tGzSsForSlabFC(end+1) = tSlabRephStartForFC; %#ok<SAGROW>
    aGzSsForSlabFC(end+1) = 0; %#ok<SAGROW>
end
[gzSsLateM0, gzSsLateM1AtRephStart] = continuousMomentFromPolylineWindow( ...
    tGzSsForSlabFC, aGzSsForSlabFC, ...
    tRfCenterInBlockForSlabFC, tSlabRephStartForFC, tSlabRephStartForFC);
if isUseSlabRephFCPreview
    gzSlabRephM0Target = -gzSsLateM0;
    gzSlabRephM1TargetAtStart = -gzSsLateM1AtRephStart;
else
    % Preserve the original Pulseq slab rephaser when slab FC is disabled.
    gzSlabRephM0Target = gz_ssReph.area;
    gzSlabRephM1TargetAtStart = 0;
end

if isUseSlabRephFCPreview
    T0Slab = ceil(mr.calcDuration(gz_ssReph) / sys.gradRasterTime) * sys.gradRasterTime;
    [gzSlabRephNatural, gzSlabFCNaturalInfo] = designMinDurationM0M1LobeRefStart( ...
        'z', gzSlabRephM0Target, gzSlabRephM1TargetAtStart, T0Slab, sys_lowPNS);
    gzSlabRephNaturalDur = ceil(mr.calcDuration(gzSlabRephNatural) / sys.gradRasterTime) * sys.gradRasterTime;
else
    gzSlabRephNatural = mr.makeTrapezoid('z', 'Area', gzSlabRephM0Target, ...
        'system', sys_lowPNS);
    gzSlabFCNaturalInfo = struct('active', false);
    gzSlabRephNaturalDur = ceil(mr.calcDuration(gzSlabRephNatural) / sys.gradRasterTime) * sys.gradRasterTime;
end

gyPreMaxNatural = mr.makeTrapezoid('y', 'Area', max(abs(phaseAreasY(:))), ...
    'system', sys_lowPNS);
gzPreMaxNatural = mr.makeTrapezoid('z', 'Area', max(abs(phaseAreasZ(:) + gzSlabRephM0Target)), ...
    'system', sys_lowPNS);

% If merged initial y/z FC is active, the corresponding PE/PAR prephaser is
% removed from the post-RF ordinary prephaser block and moved into the
% right-edge-aligned initial prep/FC module. The post-RF block then only needs
% to reserve time for the axes that are not merged plus the slab rephaser.
isUseInitialPEYFCPreview = isUseFlowComp && isFCInitialPEY;
isUseInitialPARZFCPreview = isUseFlowComp && isFCInitialPARZ;
postRfPrephNaturalDurs = [];
if ~isUseInitialPEYFCPreview
    postRfPrephNaturalDurs(end+1) = mr.calcDuration(gyPreMaxNatural); %#ok<SAGROW>
end
if isUseInitialPARZFCPreview
    postRfPrephNaturalDurs(end+1) = gzSlabRephNaturalDur; %#ok<SAGROW>
else
    postRfPrephNaturalDurs(end+1) = mr.calcDuration(gzPreMaxNatural); %#ok<SAGROW>
end
naturalGPEPreDur = ceil(max(postRfPrephNaturalDurs) / sys.gradRasterTime) * sys.gradRasterTime;

switch lower(gPEPreTimingMode)
    case {'shortest', 'minimum', 'min'}
        gPEPreDur = naturalGPEPreDur;
        gPEPreModeTag = 'PEpreMin';
    case {'fixed', 'tpre'}
        gPEPreDur = round(fixedPEPreDuration / sys.gradRasterTime) * sys.gradRasterTime;
        if gPEPreDur + sys.gradRasterTime/10 < naturalGPEPreDur
            error(['fixedPEPreDuration = %.6f ms is shorter than the shortest feasible ', ...
                   'PE/slab prephaser duration %.6f ms. Increase fixedPEPreDuration or use shortest mode.'], ...
                   gPEPreDur*1e3, naturalGPEPreDur*1e3);
        end
        gPEPreModeTag = sprintf('PEpreFixed%gus', round(gPEPreDur*1e6));
    otherwise
        error('Unknown gPEPreTimingMode = %s. Use ''shortest'' or ''fixed''.', gPEPreTimingMode);
end
if isUseSlabRephFCPreview
    % The H/B slab-FC helper uses T = 4*r, so reserve a 4-raster duration.
    gPEPreDur = ceil(gPEPreDur / (4*sys.gradRasterTime)) * (4*sys.gradRasterTime);
end

gyPreMaxDurForObj = max(gPEPreDur, ceil(mr.calcDuration(gyPreMaxNatural) / sys.gradRasterTime) * sys.gradRasterTime);
gyPreMax = mr.makeTrapezoid('y', 'Area', max(abs(phaseAreasY(:))), ...
    'Duration', gyPreMaxDurForObj, 'system', sys_lowPNS);
if isUseInitialPARZFCPreview
    if isUseSlabRephFCPreview
        [gzPreMax, gzSlabFCPreviewInfo] = makeM0M1LobeRefStart( ...
            'z', gzSlabRephM0Target, gzSlabRephM1TargetAtStart, gPEPreDur, sys_lowPNS);
    else
        gzPreMax = mr.makeTrapezoid('z', 'Area', gzSlabRephM0Target, ...
            'Duration', gPEPreDur, 'system', sys_lowPNS);
        gzSlabFCPreviewInfo = struct('active', false);
    end
else
    gzPreMax = mr.makeTrapezoid('z', 'Area', max(abs(phaseAreasZ(:) + gzSlabRephM0Target)), ...
        'Duration', gPEPreDur, 'system', sys_lowPNS);
    gzSlabFCPreviewInfo = struct('active', false);
end
fprintf(['PE/slab prephaser timing mode: %s\n', ...
         '  shortest feasible duration = %.6f ms\n', ...
         '  selected common duration   = %.6f ms\n', ...
         '  fixedPEPreDuration input   = %.6f ms\n', ...
         '  TxPre remains %.6f ms for spoilers/fixed PE timing, but x/cos prep uses min common duration.\n'], ...
    gPEPreTimingMode, naturalGPEPreDur*1e3, gPEPreDur*1e3, fixedPEPreDuration*1e3, TxPre*1e3);
if isUseSlabRephFCPreview
    fprintf(['  Step 9d slab FC active: gz_ss late M0/M1 about rephaser start = %.9g / %.9g; ', ...
             'slab target M0/M1 = %.9g / %.9g; slab FC duration = %.6f ms'], ...
        gzSsLateM0, gzSsLateM1AtRephStart, gzSlabRephM0Target, gzSlabRephM1TargetAtStart, gPEPreDur*1e3);
end


%% Wave gradients: y sine, z cosine

% One wave object per echo allows the cosine z pre-blip to compensate the
% accumulated area of prior cosine wave/blip objects. The cosine wave is
% designed first because it has a nonzero endpoint and therefore determines
% the minimum slew-safe post-ramp duration. The sine wave then uses the same
% block envelope duration, with zero padding after its ADC-synchronized part.
gyWave = cell(1, Nechoes);
gzWave = cell(1, Nechoes);
gzCosPre = cell(1, Nechoes);
waveAreaY = zeros(1, Nechoes);
waveAreaZ = zeros(1, Nechoes);
sineReadM1 = zeros(1, Nechoes);  % M1 of ADC-window sine wave only, no pre/post padding
cosRampUpM1 = zeros(1, Nechoes); % M1 of cosine ramp-up only, reference = readout-module start
cosInitialFCTargetM1 = 0;
cosTiming = cell(1, Nechoes);

prevCosArea = 0;
cosWaveBlockDurMax = readoutModuleDurX;
for c = 1:Nechoes
    if isUseWave_cos
        debugThis = waveDebugFlag;
        [gzCosPre{c}, gzWave{c}, waveAreaZ(c), prevCosArea, cosTiming{c}, cosRampUpM1(c)] = defineCosineReadoutWave( ...
            waveCosChannel, Tread, T_cosPreTrap, T_wavePrePad, readoutModuleDurX, ...
            sys, sys_lowPNS, Ncycles, gwave_max, swave_max, physical_slew_max, ...
            adc, c, prevCosArea, waveInfoFlag, debugThis);
        % gzCosPre is rebuilt below with the final common prep duration.
        gzWave{c}.id = seq.registerGradEvent(gzWave{c});
        cosWaveBlockDurMax = max(cosWaveBlockDurMax, mr.calcDuration(gzWave{c}));
    end
end

% Use the cosine-derived block envelope for the sine wave so the post-ADC
% region is consistent across y/z waves. If cosine is disabled, this simply
% falls back to the x-readout module duration.
waveBlockEnvelopeDur = cosWaveBlockDurMax;
for c = 1:Nechoes
    if isUseWave_sin
        debugThis = waveDebugFlag && (c == 1);
        [gyWave{c}, waveAreaY(c), sineReadM1(c)] = defineSineReadoutWave( ...
            waveSinChannel, Tread, T_wavePrePad, waveBlockEnvelopeDur, ...
            sys, Ncycles, gwave_max, swave_max, physical_slew_max, ...
            adc, c, waveInfoFlag && (c == 1), debugThis);
        gyWave{c}.id = seq.registerGradEvent(gyWave{c});
    end
end

% Optional y-axis FC/prephaser modules.
%
% Step 3: optional inter-echo sine FC remains a zero-area four-point lobe.
% Step 9 revised: when PE-y FC is enabled, merge the echo-1 y sine/PE-flow
% compensation into the PE-y prephaser itself. Then the initial y object has
% the desired PE M0 and an M1 chosen to cancel the sine/wave contribution at
% TE1. This avoids a separate zero-M0 PE-FC bipolar and keeps the y module
% glued to the echo-1 readout prep module.
isUseInitialSineFC = isUseFlowComp && isFCInitialSine && isUseWave_sin;
isUseInitialPEYFC = isUseFlowComp && isFCInitialPEY;
isUseInitialPARZFC = isUseFlowComp && isFCInitialPARZ;
isUseSlabRephFC = isUseFlowComp && isFCSlabRephZ;
isUseMergedInitialYPreFC = isUseInitialPEYFC;     % Step 9 revised: PE-y M0/M1-designed prephaser per LIN index
isUseMergedInitialZPreFC = isUseInitialPARZFC;    % Step 9c: z/PAR M0/M1-designed prephaser merged with cosine FC
isUseSeparateInitialSineFC = isUseInitialSineFC && ~isUseMergedInitialYPreFC;
isUseInitialYFC = isUseSeparateInitialSineFC || isUseMergedInitialYPreFC;
isUseInitialZFC = isUseMergedInitialZPreFC;
isUseInterEchoSineFC = isUseFlowComp && isFCInterEchoSine && isUseWave_sin && (Nechoes > 1);
isUseInterEchoPEYFC = isUseFlowComp && isFCInterEchoPEY && (Nechoes > 1);
isUseInterEchoPARZFC = isUseFlowComp && isFCInterEchoPARZ && (Nechoes > 1);
isUseInterEchoYFC = isUseInterEchoSineFC || isUseInterEchoPEYFC;
isUseInterEchoZFC = isUseInterEchoPARZFC;
isUseInterEchoCosEndpointM1Correction = ...
    isUseInterEchoZFC && isUseWave_cos && isApplyInterEchoCosEndpointM1Correction;
isUseAnySineFC = isUseInitialYFC || isUseInterEchoYFC;
isUseInitialReadoutFC = isUseFlowComp && isFCInitialReadout;
isUseInitialCosineFC = isUseFlowComp && isFCInitialCosine && isUseWave_cos;
isUseSeparateInitialCosineFC = isUseInitialCosineFC && ~isUseMergedInitialZPreFC;
isUseAnyInitialXCosFC = isUseInitialReadoutFC || isUseInitialCosineFC || isUseMergedInitialZPreFC;
gySineFC = cell(1, Nechoes);              % echo-indexed zero-M0 y FC for separate sine/inter-echo use
gyInitialFCByY = cell(1, Ny);             % echo-1 PE-index-dependent y prep/FC objects
gySineFCNaturalDur = zeros(1, Nechoes);
gySineFCM0 = zeros(1, Nechoes);
gySineFCM1 = zeros(1, Nechoes);
gySineFCTargetM1 = zeros(1, Nechoes);     % pure sine targets for reference/inter-echo use
gySineFCRampTime = zeros(1, Nechoes);
gySineFCGpeak = zeros(1, Nechoes);
gySineFCSlewPeak = zeros(1, Nechoes);

% Step 9 revised diagnostics/targets for merged initial PE-y flow compensation.
gyInitialPEPreM1 = zeros(1, Ny);           % old separate-PE diagnostic, kept for reference/definitions
gyInitialPEFCTargetM1 = zeros(1, Ny);      % old separate zero-M0 target, kept for reference/definitions
gyInitialFCTargetM1ByY = zeros(1, Ny);     % helper-reference target used to build gyInitialFCByY
gyInitialFCScaleByY = zeros(1, Ny);        % only used for separate zero-M0 fallback, kept for definitions
gyInitialFCTargetM1Max = 0;
gyInitialFCMaxIdx = floor(Ny/2) + 1;
gyInitialFCMaxInfo = struct('active', false);
gyInitialMergedPreM0ByY = zeros(1, Ny);
gyInitialMergedTargetM1AboutTE1ByY = zeros(1, Ny);
gyInitialMergedTargetM1ByY = zeros(1, Ny);
gyInitialMergedFinalM0ByY = zeros(1, Ny);
gyInitialMergedFinalM1ByY = zeros(1, Ny);
gyInitialMergedFinalM1AboutTE1ByY = zeros(1, Ny);
gyInitialMergedGradPeakByY = zeros(1, Ny);
gyInitialMergedSlewPeakByY = zeros(1, Ny);

% Step 10 diagnostics/targets for inter-echo PE-y + sine FC.
gyInterEchoFCByY = cell(Nechoes, Ny);             % c,iy zero-M0 y FC for echoes 2+
gyInterEchoFCTargetM1ByY = zeros(Nechoes, Ny);   % total target = sine target + PE-y drift target
gyInterEchoPEYM1TargetByY = zeros(Nechoes, Ny);  % PE-y-only drift target
gyInterEchoFCScaleByY = zeros(Nechoes, Ny);
gyInterEchoFCTargetM1MaxByEcho = zeros(1, Nechoes);
gyInterEchoFCMaxIdxByEcho = ones(1, Nechoes);
gyInterEchoFCMaxInfoByEcho = cell(1, Nechoes);
gyInterEchoFCM1ByY = zeros(Nechoes, Ny);
gyInterEchoFCM0ByY = zeros(Nechoes, Ny);

% Step 11c: robust initial cosine-head M1 plus Step 11b inter-echo PAR-z + cosine FC.
% Step 11b diagnostics/targets for inter-echo PAR-z + cosine FC.
% The cosine target is computed from the actual previous-echo cosine tail
% plus next-echo cosine head, both referenced to the next echo center.
gzInterEchoFCByZ = cell(Nechoes, Nz);                    % c,iz z/cos FC for echoes 2+
gzInterEchoCosM0ByEcho = zeros(1, Nechoes);              % nonzero cosine prewinder M0 target
gzInterEchoCosExternalM0ByEcho = zeros(1, Nechoes);      % actual tail+head cosine M0 between echoes
gzInterEchoCosExternalM1AboutTEByEcho = zeros(1, Nechoes); % actual tail+head cosine M1 about next TE
gzInterEchoCosTailM0ByEcho = zeros(1, Nechoes);
gzInterEchoCosTailM1AboutTEByEcho = zeros(1, Nechoes);
gzInterEchoCosHeadM0ByEcho = zeros(1, Nechoes);
gzInterEchoCosHeadM1AboutTEByEcho = zeros(1, Nechoes);
gzInterEchoCosM0ClosureErrorByEcho = zeros(1, Nechoes);  % external M0 + lobe M0
gzInterEchoCosEndpointM1CorrectionByEcho = zeros(1, Nechoes); % signed +G0*dt*ESP/4 correction about next TE
gzInterEchoCosTargetM1AboutTEByEcho = zeros(1, Nechoes); % lobe target that cancels external cosine M1
gzInterEchoCosTargetM1ByEcho = zeros(1, Nechoes);        % same target about next readout-module start
gzInterEchoPARZM1TargetByZ = zeros(Nechoes, Nz);         % PAR-z-only drift target = M0_PAR*ESP
gzInterEchoFCTargetM1AboutTEByZ = zeros(Nechoes, Nz);    % total target about next echo center
gzInterEchoFCTargetM1ByZ = zeros(Nechoes, Nz);           % total helper-reference target
gzInterEchoFinalM0ByZ = zeros(Nechoes, Nz);
gzInterEchoFinalM1ByZ = zeros(Nechoes, Nz);
gzInterEchoFinalM1AboutTEByZ = zeros(Nechoes, Nz);
gzInterEchoGradPeakByZ = zeros(Nechoes, Nz);
gzInterEchoSlewPeakByZ = zeros(Nechoes, Nz);
gzInterEchoFCMaxIdxByEcho = ones(1, Nechoes);
gzInterEchoFCTargetM1MaxByEcho = zeros(1, Nechoes);
gzInterEchoFCMaxInfoByEcho = cell(1, Nechoes);
gzInterEchoPostGapToReadoutByEcho = zeros(1, Nechoes);

% Step 9c diagnostics/targets for merged initial z/PAR + cosine flow compensation.
gzInitialFCByZ = cell(1, Nz);
gzInitialMergedPreM0ByZ = zeros(1, Nz);
gzInitialMergedCosM0 = 0;
gzInitialMergedCosTargetM1 = 0;  % robust cosine-only target, reference = readout-module start
gzInitialCosHeadM0 = 0;
gzInitialCosHeadM1AboutTE1 = 0;
gzInitialCosHeadM1AboutReadoutStart = 0;
gzInitialCosM0ClosureError = 0;
gzInitialMergedPredictedTotalM1AtTE1ByZ = zeros(1, Nz);
gzInitialMergedTargetM1AboutTE1ByZ = zeros(1, Nz);
gzInitialMergedTargetM1ByZ = zeros(1, Nz);
gzInitialMergedFinalM0ByZ = zeros(1, Nz);
gzInitialMergedFinalM1ByZ = zeros(1, Nz);
gzInitialMergedFinalM1AboutTE1ByZ = zeros(1, Nz);
gzInitialMergedGradPeakByZ = zeros(1, Nz);
gzInitialMergedSlewPeakByZ = zeros(1, Nz);
gzInitialFCMaxIdx = floor(Nz/2) + 1;
gzInitialFCTargetM1Max = 0;
gzInitialFCMaxInfo = struct('active', false);
gzSlabFCInfo = gzSlabFCNaturalInfo;

initialReadoutFCInfo = struct('active', false);
initialCosineFCInfo = struct('active', false);
initialXCosFCGrowIters = 0;

if isUseInitialSineFC
    gySineFCTargetM1(1) = -0.5 * sineReadM1(1);
end

if isUseMergedInitialYPreFC
    % Merged echo-1 y prephaser target.
    % Desired about TE1: M1_y_pre(TE1 ref) cancels the sine readout target.
    % Helper reference is readout-module start. Because M0 is nonzero,
    % convert by adding t_adc_center*M0:
    %   M1_about_readoutStart = M1_about_TE1 + t_adc_center*M0.
    gyInitialMergedPreM0ByY = phaseAreasY;
    gyInitialMergedTargetM1AboutTE1ByY = gySineFCTargetM1(1) * ones(1, Ny);
    gyInitialMergedTargetM1ByY = gyInitialMergedTargetM1AboutTE1ByY + t_adc_center * gyInitialMergedPreM0ByY;
    gyInitialFCTargetM1ByY = gyInitialMergedTargetM1ByY;
    [~, gyInitialFCMaxIdx] = max(abs(gyInitialFCTargetM1ByY));
    gyInitialFCTargetM1Max = gyInitialFCTargetM1ByY(gyInitialFCMaxIdx);
elseif isUseSeparateInitialSineFC
    % Fallback for sine-only initial FC without PE-y merging. This is the old
    % zero-M0 design/scaling path.
    gyInitialFCTargetM1ByY(:) = gySineFCTargetM1(1);
    [~, gyInitialFCMaxIdx] = max(abs(gyInitialFCTargetM1ByY));
    gyInitialFCTargetM1Max = gyInitialFCTargetM1ByY(gyInitialFCMaxIdx);
    [~, sineFCTimingMin] = makeZeroAreaM1FourPointFC( ...
        waveSinChannel, gyInitialFCTargetM1Max, NaN, sys_lowPNS2);
    gySineFCNaturalDur(1) = sineFCTimingMin.T;
end

% Robust echo-1 cosine-head moments from the actual arbitrary-gradient
% waveform. The integration window is readout-module start through the
% first ADC center, with M1 reported about both TE1 and module start.
if isUseWave_cos
    [gzInitialCosHeadM0, gzInitialCosHeadM1AboutTE1, ...
        gzInitialCosHeadM1AboutReadoutStart] = ...
        calcInitialCosHeadMoments(cosTiming{1}, t_adc_center);

    if isUseInitialCosineFC
        % The cosine-only prep lobe must cancel the actual cosine head at
        % TE1. Convert that target to the helper reference at readout start.
        cosInitialFCTargetM1 = ...
            -gzInitialCosHeadM1AboutTE1 ...
            + t_adc_center * cosTiming{1}.preTrapAreaTarget;
    else
        cosInitialFCTargetM1 = 0;
    end
else
    gzInitialCosHeadM0 = 0;
    gzInitialCosHeadM1AboutTE1 = 0;
    gzInitialCosHeadM1AboutReadoutStart = 0;
    cosInitialFCTargetM1 = 0;
end

if isUseMergedInitialZPreFC
    % Merged echo-1 z prephaser target.
    % Split original gzPreComb into:
    %   post-RF: slab rephaser only (gzSlabReph)
    %   initial prep: z/PAR prephaser + cosine prewinder/FC
    % Desired about TE1: PAR component has zero M1 about TE1. Convert to
    % helper/readout-start reference by adding t_adc_center*M0_PAR.
    if isUseWave_cos
        gzInitialMergedCosM0 = cosTiming{1}.preTrapAreaTarget;
        gzInitialMergedCosTargetM1 = cosInitialFCTargetM1;
        gzInitialCosM0ClosureError = ...
            gzInitialMergedCosM0 + gzInitialCosHeadM0;
    else
        gzInitialMergedCosM0 = 0;
        gzInitialMergedCosTargetM1 = 0;
        gzInitialCosM0ClosureError = 0;
    end

    gzInitialMergedPreM0ByZ = phaseAreasZ + gzInitialMergedCosM0;

    % The helper reference is the readout-module start. The robust
    % cosine-only helper target already includes the M0_cos translation.
    % Add only the PAR M0 translation for each index.
    gzInitialMergedTargetM1ByZ = ...
        gzInitialMergedCosTargetM1 + t_adc_center * phaseAreasZ;

    % Convert the complete merged-lobe target back to TE1 for reporting.
    gzInitialMergedTargetM1AboutTE1ByZ = ...
        gzInitialMergedTargetM1ByZ ...
        - t_adc_center * gzInitialMergedPreM0ByZ;

    [~, gzInitialFCMaxIdx] = max(abs(gzInitialMergedTargetM1ByZ));
    gzInitialFCTargetM1Max = gzInitialMergedTargetM1ByZ(gzInitialFCMaxIdx);
end

if isUseInterEchoYFC
    if isUseInterEchoSineFC
        sineFCTargetM1 = -sineReadM1(1);
    else
        sineFCTargetM1 = 0;
    end
    for c = 2:Nechoes
        gySineFCTargetM1(c) = sineFCTargetM1;
        esp_c = TE(c) - TE(c-1);
        if isUseInterEchoPEYFC
            % If echo c-1 is already FC'ed, the maintained PE zeroth moment
            % accumulates an additional M1_PE = -M0_PE*ESP at echo c.
            % The zero-M0 inter-echo y FC lobe therefore targets +M0_PE*ESP.
            gyInterEchoPEYM1TargetByY(c, :) = phaseAreasY * esp_c;
        end
        gyInterEchoFCTargetM1ByY(c, :) = sineFCTargetM1 + gyInterEchoPEYM1TargetByY(c, :);
        if isUseInterEchoPEYFC
            [~, gyInterEchoFCMaxIdxByEcho(c)] = max(abs(gyInterEchoFCTargetM1ByY(c, :)));
            gyInterEchoFCTargetM1MaxByEcho(c) = gyInterEchoFCTargetM1ByY(c, gyInterEchoFCMaxIdxByEcho(c));
            targetForNatural = gyInterEchoFCTargetM1MaxByEcho(c);
        else
            targetForNatural = gySineFCTargetM1(c);
        end
        [~, sineFCTimingMin] = makeZeroAreaM1FourPointFC( ...
            waveSinChannel, targetForNatural, NaN, sys_lowPNS2);
        gySineFCNaturalDur(c) = sineFCTimingMin.T;
    end
end

if isUseInterEchoZFC
    for c = 2:Nechoes
        esp_c = TE(c) - TE(c-1);
        if isUseWave_cos
            gzInterEchoCosM0ByEcho(c) = cosTiming{c}.preTrapAreaTarget;

            % Robust cosine M1 target: integrate the actually constructed
            % previous-echo cosine tail and next-echo cosine head over
            % [TE(c-1), TE(c)], with the next echo center as tRef = 0.
            [gzInterEchoCosExternalM0ByEcho(c), ...
             gzInterEchoCosExternalM1AboutTEByEcho(c), ...
             gzInterEchoCosTailM0ByEcho(c), ...
             gzInterEchoCosTailM1AboutTEByEcho(c), ...
             gzInterEchoCosHeadM0ByEcho(c), ...
             gzInterEchoCosHeadM1AboutTEByEcho(c)] = ...
                calcInterEchoCosExternalMoments( ...
                    cosTiming{c-1}, cosTiming{c}, esp_c, t_adc_center);

            % The endpoint-inclusive cosine construction repeats G0 at the
            % ramp/read-wave joins. Relative to the continuous ideal waveform,
            % this behaves like an extra half-raster area G0*dt/2 whose
            % effective centroid is ESP/2 before the next ADC center. Therefore
            % the observed residual is -G0*dt*ESP/4 about TE(c), and the H/B
            % lobe receives the opposite signed correction +G0*dt*ESP/4.
            %
            % G0 is already in Pulseq units (Hz/m), so do not multiply by gamma.
            if isUseInterEchoCosEndpointM1Correction
                G0_endpoint = cosTiming{c}.G0;
                gzInterEchoCosEndpointM1CorrectionByEcho(c) = ...
                    G0_endpoint * sys.gradRasterTime * esp_c / 4;
            else
                gzInterEchoCosEndpointM1CorrectionByEcho(c) = 0;
            end

            % The inter-echo H/B lobe cancels the actual external cosine M1
            % about the next ADC center and, optionally, adds the analytic
            % half-raster endpoint correction above. Convert this nonzero-M0
            % lobe target to the helper reference (next readout-module start).
            gzInterEchoCosTargetM1AboutTEByEcho(c) = ...
                -gzInterEchoCosExternalM1AboutTEByEcho(c) ...
                - gzInterEchoCosEndpointM1CorrectionByEcho(c);
            gzInterEchoCosTargetM1ByEcho(c) = ...
                gzInterEchoCosTargetM1AboutTEByEcho(c) ...
                + t_adc_center * gzInterEchoCosM0ByEcho(c);

            % This should remain near zero because the existing cosine M0
            % target is retained; report/store any mismatch for inspection.
            gzInterEchoCosM0ClosureErrorByEcho(c) = ...
                gzInterEchoCosExternalM0ByEcho(c) + gzInterEchoCosM0ByEcho(c);
        else
            gzInterEchoCosM0ByEcho(c) = 0;
            gzInterEchoCosTargetM1AboutTEByEcho(c) = 0;
            gzInterEchoCosTargetM1ByEcho(c) = 0;
        end

        % If echo c-1 has zero PAR first moment, translating the maintained
        % PAR zeroth moment to echo c gives -M0_PAR*ESP. The correction module
        % therefore adds +M0_PAR*ESP. This term is added about TE(c); the
        % subsequent helper-reference conversion uses the total lobe M0.
        gzInterEchoPARZM1TargetByZ(c, :) = phaseAreasZ * esp_c;
        gzInterEchoFCTargetM1AboutTEByZ(c, :) = ...
            gzInterEchoCosTargetM1AboutTEByEcho(c) ...
            + gzInterEchoPARZM1TargetByZ(c, :);
        gzInterEchoFCTargetM1ByZ(c, :) = ...
            gzInterEchoFCTargetM1AboutTEByZ(c, :) ...
            + t_adc_center * gzInterEchoCosM0ByEcho(c);

        [~, gzInterEchoFCMaxIdxByEcho(c)] = max(abs(gzInterEchoFCTargetM1ByZ(c, :)));
        gzInterEchoFCTargetM1MaxByEcho(c) = ...
            gzInterEchoFCTargetM1ByZ(c, gzInterEchoFCMaxIdxByEcho(c));
    end
end

if isUseInterEchoZFC && isUseWave_cos
    fprintf('\nStep 11b robust inter-echo cosine M1 targets (reference = next echo center):\n');
    for c = 2:Nechoes
        fprintf(['  Echo %d: external cos M0/M1 = %+.9g / %+.9g, ', ...
                 'endpoint dM1 = %+.9g, lobe cos M0/M1 target = %+.9g / %+.9g, ', ...
                 'M0 closure = %+.3e\n'], ...
            c, gzInterEchoCosExternalM0ByEcho(c), ...
            gzInterEchoCosExternalM1AboutTEByEcho(c), ...
            gzInterEchoCosEndpointM1CorrectionByEcho(c), ...
            gzInterEchoCosM0ByEcho(c), ...
            gzInterEchoCosTargetM1AboutTEByEcho(c), ...
            gzInterEchoCosM0ClosureErrorByEcho(c));
    end
end

% Build the x dephaser/flyback, cosine prewinder, and optional sine FC lobe
% as a common-duration prep/FC module. For each echo, first find the shortest
% feasible duration of every active component, then rebuild all active
% components with the larger duration. Echo 1 can include the Step-4 initial sine FC
% and Step-6 initial x/cos M1-corrected M0-preserving lobes.
gxPrep = cell(1, Nechoes);
gxPrepNaturalDur = zeros(1, Nechoes);
cosPreNaturalDur = zeros(1, Nechoes);
prepModuleDurTarget = zeros(1, Nechoes);
for c = 1:Nechoes
    if c == 1
        gxPrepArea = gxPrepAreaEcho1;
        gxPrepNatural = groPreNatural;
    else
        gxPrepArea = gxPrepAreaRest;
        gxPrepNatural = groFlyBackNatural;
    end
    gxPrepNaturalDur(c) = ceil(mr.calcDuration(gxPrepNatural) / sys.gradRasterTime) * sys.gradRasterTime;

    if isUseWave_cos
        cosPreNaturalDur(c) = ceil(cosTiming{c}.preTrapNaturalDur / sys.gradRasterTime) * sys.gradRasterTime;
    else
        cosPreNaturalDur(c) = 0;
    end

    prepModuleDurTarget(c) = max([gxPrepNaturalDur(c), cosPreNaturalDur(c), gySineFCNaturalDur(c)]);
    prepModuleDurTarget(c) = ceil(prepModuleDurTarget(c) / sys.gradRasterTime) * sys.gradRasterTime;
    if ((isUseInitialYFC || isUseInitialZFC) && c == 1) || ((isUseInterEchoYFC || isUseInterEchoZFC) && c > 1) || (isUseAnyInitialXCosFC && c == 1)
        % Four-point bipolar overlays use T = 4*r; keep the common module
        % duration compatible with that structure, then rebuild all components.
        prepModuleDurTarget(c) = ceil(prepModuleDurTarget(c) / (4*sys.gradRasterTime)) * (4*sys.gradRasterTime);
    end

    if c == 1 && (isUseAnyInitialXCosFC || isUseMergedInitialYPreFC || isUseMergedInitialZPreFC)
        Ttry = prepModuleDurTarget(c);
        maxIter = 1000;
        okInitial = false;
        lastErr = '';
        for iter = 1:maxIter
            Ttry = ceil(Ttry / (4*sys.gradRasterTime)) * (4*sys.gradRasterTime);
            try
                if isUseInitialReadoutFC
                    [gxCandidate, gxInfo] = makeM0PreservingM1CorrectedLobe( ...
                        ax.d1, gxPrepArea, readoutInitialFCTargetM1, Ttry, prepToReadoutGap, sys_lowPNS2);
                else
                    gxCandidate = mr.makeTrapezoid(ax.d1, 'Area', gxPrepArea, ...
                        'Duration', Ttry, 'system', sys_lowPNS);
                    gxInfo = struct('active', false);
                end

                if c == 1 && isUseMergedInitialZPreFC
                    gzMergedCandidateByZ = cell(1, Nz);
                    gzMergedInfoByZ = cell(1, Nz);
                    for izPARFC = 1:Nz
                        [gzMergedCandidateByZ{izPARFC}, gzMergedInfoByZ{izPARFC}] = makeM0PreservingM1CorrectedLobe(waveCosChannel, gzInitialMergedPreM0ByZ(izPARFC), ...
                            gzInitialMergedTargetM1ByZ(izPARFC), Ttry, prepToReadoutGap, sys_lowPNS2);
                    end
                    gzCandidate = [];
                    gzInfo = struct('active', false);
                elseif isUseWave_cos
                    if isUseSeparateInitialCosineFC
                        [gzCandidate, gzInfo] = makeM0PreservingM1CorrectedLobe( ...
                            waveCosChannel, cosTiming{c}.preTrapAreaTarget, ...
                            cosInitialFCTargetM1, Ttry, prepToReadoutGap, sys_lowPNS2);
                    else
                        gzCandidate = mr.makeTrapezoid(waveCosChannel, ...
                            'Area', cosTiming{c}.preTrapAreaTarget, ...
                            'Duration', Ttry, 'system', sys_lowPNS);
                        gzInfo = struct('active', false);
                    end
                else
                    gzInfo = struct('active', false);
                end

                if c == 1 && isUseMergedInitialYPreFC
                    gyMergedCandidateByY = cell(1, Ny);
                    gyMergedInfoByY = cell(1, Ny);
                    for iyPEFC = 1:Ny
                        [gyMergedCandidateByY{iyPEFC}, gyMergedInfoByY{iyPEFC}] = makeM0PreservingM1CorrectedLobe( ...
                            waveSinChannel, gyInitialMergedPreM0ByY(iyPEFC), ...
                            gyInitialMergedTargetM1ByY(iyPEFC), Ttry, prepToReadoutGap, sys_lowPNS2);
                    end
                elseif c == 1 && isUseSeparateInitialSineFC
                    [gyCandidate, sineFCTiming] = makeZeroAreaM1FourPointFC( ...
                        waveSinChannel, gyInitialFCTargetM1Max, Ttry, sys_lowPNS2);
                elseif c > 1 && isUseInterEchoPEYFC
                    [gyCandidate, sineFCTiming] = makeZeroAreaM1FourPointFC( ...
                        waveSinChannel, gyInterEchoFCTargetM1MaxByEcho(c), Ttry, sys_lowPNS2);
                elseif c > 1 && isUseInterEchoSineFC
                    [gyCandidate, sineFCTiming] = makeZeroAreaM1FourPointFC( ...
                        waveSinChannel, gySineFCTargetM1(c), Ttry, sys_lowPNS2);
                end

                okInitial = true;
                break;
            catch ME
                lastErr = ME.message;
                Ttry = Ttry + 4*sys.gradRasterTime;
            end
        end
        if ~okInitial
            error('Could not design initial x/cos/sine FC prep module after %d iterations. Last error: %s', maxIter, lastErr);
        end

        prepModuleDurTarget(c) = Ttry;
        initialXCosFCGrowIters = iter - 1;
        gxPrep{c} = gxCandidate;
        initialReadoutFCInfo = gxInfo;

        if c == 1 && isUseMergedInitialZPreFC
            gzInitialFCByZ = gzMergedCandidateByZ;
            for izPARFC = 1:Nz
                gzInitialMergedFinalM0ByZ(izPARFC) = gzMergedInfoByZ{izPARFC}.finalM0;
                gzInitialMergedFinalM1ByZ(izPARFC) = gzMergedInfoByZ{izPARFC}.finalM1;
                gzInitialMergedFinalM1AboutTE1ByZ(izPARFC) = ...
                    gzMergedInfoByZ{izPARFC}.finalM1 ...
                    - t_adc_center * gzMergedInfoByZ{izPARFC}.finalM0;
                gzInitialMergedPredictedTotalM1AtTE1ByZ(izPARFC) = ...
                    gzInitialMergedFinalM1AboutTE1ByZ(izPARFC) ...
                    + gzInitialCosHeadM1AboutTE1;
                gzInitialMergedGradPeakByZ(izPARFC) = gzMergedInfoByZ{izPARFC}.gradPeak;
                gzInitialMergedSlewPeakByZ(izPARFC) = gzMergedInfoByZ{izPARFC}.slewPeak;
            end
            gzInitialFCMaxInfo = gzMergedInfoByZ{gzInitialFCMaxIdx};
            if isUseWave_cos
                % Wave carry should include only the cosine prewinder contribution,
                % not the z/PAR encoding area that is merged into gzInitialFCByZ{iz}.
                waveAreaZ(c) = gzInitialMergedCosM0 + gzWave{c}.area;
            end
        elseif isUseWave_cos
            gzCosPre{c} = gzCandidate;
            initialCosineFCInfo = gzInfo;
            waveAreaZ(c) = gzCosPre{c}.area + gzWave{c}.area;
        end

        if c == 1 && isUseMergedInitialYPreFC
            gyInitialFCByY = gyMergedCandidateByY;
            for iyPEFC = 1:Ny
                gyInitialMergedFinalM0ByY(iyPEFC) = gyMergedInfoByY{iyPEFC}.finalM0;
                gyInitialMergedFinalM1ByY(iyPEFC) = gyMergedInfoByY{iyPEFC}.finalM1;
                gyInitialMergedFinalM1AboutTE1ByY(iyPEFC) = gyMergedInfoByY{iyPEFC}.finalM1 - t_adc_center * gyMergedInfoByY{iyPEFC}.finalM0;
                gyInitialMergedGradPeakByY(iyPEFC) = gyMergedInfoByY{iyPEFC}.gradPeak;
                gyInitialMergedSlewPeakByY(iyPEFC) = gyMergedInfoByY{iyPEFC}.slewPeak;
            end
            gyInitialFCMaxInfo = gyMergedInfoByY{gyInitialFCMaxIdx};
        elseif c == 1 && isUseSeparateInitialSineFC
            gySineFC{c} = gyCandidate;
            gySineFCM0(c) = sineFCTiming.M0;
            gySineFCM1(c) = sineFCTiming.M1;
            gySineFCRampTime(c) = sineFCTiming.r;
            gySineFCGpeak(c) = sineFCTiming.Gpeak;
            gySineFCSlewPeak(c) = sineFCTiming.slewPeak;
            gyInitialFCMaxInfo = sineFCTiming;
        elseif c > 1 && isUseInterEchoPEYFC
            gySineFC{c} = gyCandidate;  % max-|M1| template for this echo gap
            gySineFCM0(c) = sineFCTiming.M0;
            gySineFCM1(c) = sineFCTiming.M1;
            gySineFCRampTime(c) = sineFCTiming.r;
            gySineFCGpeak(c) = sineFCTiming.Gpeak;
            gySineFCSlewPeak(c) = sineFCTiming.slewPeak;
            gyInterEchoFCMaxInfoByEcho{c} = sineFCTiming;
            [gyInterEchoFCByY, gyInterEchoFCScaleByY, gyInterEchoFCM0ByY, gyInterEchoFCM1ByY] = ...
                fillInterEchoYFCScaledWaveforms(gyInterEchoFCByY, gyInterEchoFCScaleByY, ...
                    gyInterEchoFCM0ByY, gyInterEchoFCM1ByY, c, gyInterEchoFCTargetM1ByY(c, :), ...
                    gyInterEchoFCTargetM1MaxByEcho(c), sineFCTiming, waveSinChannel, sys_lowPNS2);
        elseif c > 1 && isUseInterEchoSineFC
            gySineFC{c} = gyCandidate;
            gySineFCM0(c) = sineFCTiming.M0;
            gySineFCM1(c) = sineFCTiming.M1;
            gySineFCRampTime(c) = sineFCTiming.r;
            gySineFCGpeak(c) = sineFCTiming.Gpeak;
            gySineFCSlewPeak(c) = sineFCTiming.slewPeak;
        end
    elseif c > 1 && isUseInterEchoZFC
        % Step 11b: inter-echo z/PAR + cosine FC. The M1 targets were
        % computed from the actual previous-tail/next-head cosine waveform
        % about TE(c), then converted to the next readout-start reference.
        % Search a common duration for x, y, and all z/PAR indices.
        Ttry = prepModuleDurTarget(c);
        maxIter = 1000;
        okInterEchoZ = false;
        lastErr = '';
        for iter = 1:maxIter
            Ttry = ceil(Ttry / (4*sys.gradRasterTime)) * (4*sys.gradRasterTime);
            try
                [postGapToRefTry, fcFreeGapTry] = calcInterEchoPostGapToReadoutStart( ...
                    TE(c) - TE(c-1), Ttry, tReadGradStartForFC, tReadGradEndForFC, sys.gradRasterTime);
                if fcFreeGapTry < -sys.gradRasterTime/10
                    error('Inter-echo z FC duration %.6f ms exceeds readout-gradient free gap %.6f ms.', ...
                        Ttry*1e3, (TE(c) - TE(c-1) - (tReadGradEndForFC - tReadGradStartForFC))*1e3);
                end

                gxCandidate = mr.makeTrapezoid(ax.d1, 'Area', gxPrepArea, ...
                    'Duration', Ttry, 'system', sys_lowPNS2);

                gzInterCandidateByZ = cell(1, Nz);
                gzInterInfoByZ = cell(1, Nz);
                for izPARFC = 1:Nz
                    [gzInterCandidateByZ{izPARFC}, gzInterInfoByZ{izPARFC}] = makeM0PreservingM1CorrectedLobe( ...
                        waveCosChannel, gzInterEchoCosM0ByEcho(c), ...
                        gzInterEchoFCTargetM1ByZ(c, izPARFC), Ttry, postGapToRefTry, sys_lowPNS2);
                end

                if isUseInterEchoPEYFC
                    [gyCandidate, sineFCTiming] = makeZeroAreaM1FourPointFC( ...
                        waveSinChannel, gyInterEchoFCTargetM1MaxByEcho(c), Ttry, sys_lowPNS2);
                elseif isUseInterEchoSineFC
                    [gyCandidate, sineFCTiming] = makeZeroAreaM1FourPointFC( ...
                        waveSinChannel, gySineFCTargetM1(c), Ttry, sys_lowPNS2);
                end

                okInterEchoZ = true;
                break;
            catch ME
                lastErr = ME.message;
                Ttry = Ttry + 4*sys.gradRasterTime;
            end
        end
        if ~okInterEchoZ
            error('Could not design inter-echo z/PAR+cos FC module for echo %d after %d iterations. Last error: %s', ...
                c, maxIter, lastErr);
        end

        prepModuleDurTarget(c) = Ttry;
        gxPrep{c} = gxCandidate;
        gzInterEchoFCByZ(c, :) = gzInterCandidateByZ;
        gzInterEchoPostGapToReadoutByEcho(c) = postGapToRefTry;
        for izPARFC = 1:Nz
            gzInterEchoFinalM0ByZ(c, izPARFC) = gzInterInfoByZ{izPARFC}.finalM0;
            gzInterEchoFinalM1ByZ(c, izPARFC) = gzInterInfoByZ{izPARFC}.finalM1;
            gzInterEchoFinalM1AboutTEByZ(c, izPARFC) = ...
                gzInterInfoByZ{izPARFC}.finalM1 - t_adc_center * gzInterInfoByZ{izPARFC}.finalM0;
            gzInterEchoGradPeakByZ(c, izPARFC) = gzInterInfoByZ{izPARFC}.gradPeak;
            gzInterEchoSlewPeakByZ(c, izPARFC) = gzInterInfoByZ{izPARFC}.slewPeak;
        end
        gzInterEchoFCMaxInfoByEcho{c} = gzInterInfoByZ{gzInterEchoFCMaxIdxByEcho(c)};
        if isUseWave_cos
            % The played inter-echo z object carries only the cosine prewinder M0;
            % the PAR drift term changes M1 but not net M0 or k-space carry.
            waveAreaZ(c) = gzInterEchoCosM0ByEcho(c) + gzWave{c}.area;
        end

        if isUseInterEchoPEYFC
            gySineFC{c} = gyCandidate;
            gySineFCM0(c) = sineFCTiming.M0;
            gySineFCM1(c) = sineFCTiming.M1;
            gySineFCRampTime(c) = sineFCTiming.r;
            gySineFCGpeak(c) = sineFCTiming.Gpeak;
            gySineFCSlewPeak(c) = sineFCTiming.slewPeak;
            gyInterEchoFCMaxInfoByEcho{c} = sineFCTiming;
            [gyInterEchoFCByY, gyInterEchoFCScaleByY, gyInterEchoFCM0ByY, gyInterEchoFCM1ByY] = ...
                fillInterEchoYFCScaledWaveforms(gyInterEchoFCByY, gyInterEchoFCScaleByY, ...
                    gyInterEchoFCM0ByY, gyInterEchoFCM1ByY, c, gyInterEchoFCTargetM1ByY(c, :), ...
                    gyInterEchoFCTargetM1MaxByEcho(c), sineFCTiming, waveSinChannel, sys_lowPNS2);
        elseif isUseInterEchoSineFC
            gySineFC{c} = gyCandidate;
            gySineFCM0(c) = sineFCTiming.M0;
            gySineFCM1(c) = sineFCTiming.M1;
            gySineFCRampTime(c) = sineFCTiming.r;
            gySineFCGpeak(c) = sineFCTiming.Gpeak;
            gySineFCSlewPeak(c) = sineFCTiming.slewPeak;
        end
    else
        gxPrep{c} = mr.makeTrapezoid(ax.d1, 'Area', gxPrepArea, ...
            'Duration', prepModuleDurTarget(c), 'system', sys_lowPNS);

        if c == 1 && isUseMergedInitialZPreFC
            if isUseWave_cos
                waveAreaZ(c) = gzInitialMergedCosM0 + gzWave{c}.area;
            end
        elseif isUseWave_cos
            gzCosPre{c} = mr.makeTrapezoid(waveCosChannel, ...
                'Area', cosTiming{c}.preTrapAreaTarget, ...
                'Duration', prepModuleDurTarget(c), 'system', sys_lowPNS);
            waveAreaZ(c) = gzCosPre{c}.area + gzWave{c}.area;
        end

        if c == 1 && isUseSeparateInitialSineFC
            [gySineFC{c}, sineFCTiming] = makeZeroAreaM1FourPointFC( ...
                waveSinChannel, gyInitialFCTargetM1Max, prepModuleDurTarget(c), sys_lowPNS2);
            gySineFCM0(c) = sineFCTiming.M0;
            gySineFCM1(c) = sineFCTiming.M1;
            gySineFCRampTime(c) = sineFCTiming.r;
            gySineFCGpeak(c) = sineFCTiming.Gpeak;
            gySineFCSlewPeak(c) = sineFCTiming.slewPeak;
            gyInitialFCMaxInfo = sineFCTiming;
        elseif c > 1 && isUseInterEchoPEYFC
            [gySineFC{c}, sineFCTiming] = makeZeroAreaM1FourPointFC( ...
                waveSinChannel, gyInterEchoFCTargetM1MaxByEcho(c), prepModuleDurTarget(c), sys_lowPNS2);
            gySineFCM0(c) = sineFCTiming.M0;
            gySineFCM1(c) = sineFCTiming.M1;
            gySineFCRampTime(c) = sineFCTiming.r;
            gySineFCGpeak(c) = sineFCTiming.Gpeak;
            gySineFCSlewPeak(c) = sineFCTiming.slewPeak;
            gyInterEchoFCMaxInfoByEcho{c} = sineFCTiming;
            [gyInterEchoFCByY, gyInterEchoFCScaleByY, gyInterEchoFCM0ByY, gyInterEchoFCM1ByY] = ...
                fillInterEchoYFCScaledWaveforms(gyInterEchoFCByY, gyInterEchoFCScaleByY, ...
                    gyInterEchoFCM0ByY, gyInterEchoFCM1ByY, c, gyInterEchoFCTargetM1ByY(c, :), ...
                    gyInterEchoFCTargetM1MaxByEcho(c), sineFCTiming, waveSinChannel, sys_lowPNS2);
        elseif c > 1 && isUseInterEchoSineFC
            [gySineFC{c}, sineFCTiming] = makeZeroAreaM1FourPointFC( ...
                waveSinChannel, gySineFCTargetM1(c), prepModuleDurTarget(c), sys_lowPNS2);
            gySineFCM0(c) = sineFCTiming.M0;
            gySineFCM1(c) = sineFCTiming.M1;
            gySineFCRampTime(c) = sineFCTiming.r;
            gySineFCGpeak(c) = sineFCTiming.Gpeak;
            gySineFCSlewPeak(c) = sineFCTiming.slewPeak;
        end
    end
end

% Fallback only: if echo-1 y FC is separate zero-M0 sine FC, make PE-index
% dependent copies by scaling the max-|M1| zero-M0 template. In the revised
% PE-y mode, gyInitialFCByY has already been filled with the merged M0/M1
% PE-prephaser objects above, so no scaling is used.
if isUseSeparateInitialSineFC
    if ~isfield(gyInitialFCMaxInfo, 'times')
        error('Internal error: missing initial y FC max waveform information.');
    end
    if abs(gyInitialFCTargetM1Max) < 1e-14
        gyInitialFCScaleByY(:) = 0;
    else
        gyInitialFCScaleByY = gyInitialFCTargetM1ByY / gyInitialFCTargetM1Max;
    end
    for iyPEFC = 1:Ny
        gyInitialFCByY{iyPEFC} = mr.makeExtendedTrapezoid(waveSinChannel, ...
            'times', gyInitialFCMaxInfo.times, ...
            'amplitudes', gyInitialFCScaleByY(iyPEFC) * gyInitialFCMaxInfo.amps, ...
            'system', sys_lowPNS2);
    end
end

fprintf('\nMinimum-duration prep/FC module summary:\n');
for c = 1:Nechoes
    fprintf(['  Echo %d: x natural=%.6f ms, cos natural=%.6f ms, sineFC natural=%.6f ms, ', ...
             'selected common prep/FC=%.6f ms\n'], ...
        c, gxPrepNaturalDur(c)*1e3, cosPreNaturalDur(c)*1e3, gySineFCNaturalDur(c)*1e3, prepModuleDurTarget(c)*1e3);
end
if isUseSeparateInitialSineFC || isUseInterEchoSineFC || isUseInterEchoPEYFC
    fprintf('\nY FC four-point lobe summary:\n');
    if isUseSeparateInitialSineFC
        fprintf('  Separate initial echo-1 target from -0.5*sineReadM1(1): %.9g 1/m*s\n', gySineFCTargetM1(1));
        fprintf(['  Echo 1: targetM1=%.9g, achieved M1=%.9g, residual=%.3g, ', ...
                 'M0=%.3g, common prepDur=%.6f ms, r=%.6f us, Gpeak=%.6f kHz/m, slew=%.6f T/m/s equiv\n'], ...
            gySineFCTargetM1(1), gySineFCM1(1), gySineFCM1(1)-gySineFCTargetM1(1), ...
            gySineFCM0(1), prepModuleDurTarget(1)*1e3, gySineFCRampTime(1)*1e6, ...
            gySineFCGpeak(1)*1e-3, gySineFCSlewPeak(1)/sys.gamma);
    end
    if isUseInterEchoPEYFC
        fprintf('  Inter-echo PE-y Step 10 active. Pure sine base target = %.9g 1/m*s\n', gySineFCTargetM1(2));
        for c = 2:Nechoes
            fprintf(['  Echo %d: total target range=[%.9g %.9g], max-|M1| iy=%d target=%.9g, ', ...
                     'achieved template M1=%.9g, M0=%.3g, prepDur=%.6f ms, r=%.6f us, ', ...
                     'Gpeak=%.6f kHz/m, slew=%.6f T/m/s equiv\n'], ...
                c, min(gyInterEchoFCTargetM1ByY(c, :)), max(gyInterEchoFCTargetM1ByY(c, :)), ...
                gyInterEchoFCMaxIdxByEcho(c), gyInterEchoFCTargetM1MaxByEcho(c), gySineFCM1(c), ...
                gySineFCM0(c), prepModuleDurTarget(c)*1e3, gySineFCRampTime(c)*1e6, ...
                gySineFCGpeak(c)*1e-3, gySineFCSlewPeak(c)/sys.gamma);
        end
    elseif isUseInterEchoSineFC
        fprintf('  Inter-echo target from -sineReadM1(1): %.9g 1/m*s\n', -sineReadM1(1));
        for c = 2:Nechoes
            fprintf(['  Echo %d: targetM1=%.9g, achieved M1=%.9g, residual=%.3g, ', ...
                     'M0=%.3g, common prepDur=%.6f ms, r=%.6f us, Gpeak=%.6f kHz/m, slew=%.6f T/m/s equiv\n'], ...
                c, gySineFCTargetM1(c), gySineFCM1(c), gySineFCM1(c)-gySineFCTargetM1(c), ...
                gySineFCM0(c), prepModuleDurTarget(c)*1e3, gySineFCRampTime(c)*1e6, ...
                gySineFCGpeak(c)*1e-3, gySineFCSlewPeak(c)/sys.gamma);
        end
    end
end

if isUseInitialYFC
    fprintf('\nInitial y FC / PE-y Step 9 revised summary:\n');
    fprintf('  active sine initial FC=%d, active PE-y merged prephaser FC=%d\n', ...
        isUseInitialSineFC, isUseMergedInitialYPreFC);
    fprintf('  pure sine initial target M1 = %.9g 1/m*s\n', gySineFCTargetM1(1));
    if isUseMergedInitialYPreFC
        fprintf('  merged y prephaser M0 range = [%.9g, %.9g] 1/m\n', ...
            min(gyInitialMergedPreM0ByY), max(gyInitialMergedPreM0ByY));
        fprintf('  target M1 about TE1 range = [%.9g, %.9g] 1/m*s\n', ...
            min(gyInitialMergedTargetM1AboutTE1ByY), max(gyInitialMergedTargetM1AboutTE1ByY));
        fprintf('  helper-ref target M1 range = [%.9g, %.9g] 1/m*s\n', ...
            min(gyInitialMergedTargetM1ByY), max(gyInitialMergedTargetM1ByY));
        fprintf('  final M0 range = [%.9g, %.9g] 1/m; final helper-ref M1 range = [%.9g, %.9g] 1/m*s\n', ...
            min(gyInitialMergedFinalM0ByY), max(gyInitialMergedFinalM0ByY), ...
            min(gyInitialMergedFinalM1ByY), max(gyInitialMergedFinalM1ByY));
        fprintf('  final M1 about TE1 range = [%.9g, %.9g] 1/m*s\n', ...
            min(gyInitialMergedFinalM1AboutTE1ByY), max(gyInitialMergedFinalM1AboutTE1ByY));
        fprintf(['  worst helper-ref |target M1|: iy=%d, LIN label=%d, target=%.9g 1/m*s, ', ...
                 'duration=%.6f ms, Gpk range=[%.6f %.6f] kHz/m, slew range=[%.6f %.6f] T/m/s equiv\n'], ...
            gyInitialFCMaxIdx, gyInitialFCMaxIdx-1, gyInitialFCTargetM1Max, ...
            prepModuleDurTarget(1)*1e3, min(gyInitialMergedGradPeakByY)*1e-3, ...
            max(gyInitialMergedGradPeakByY)*1e-3, min(gyInitialMergedSlewPeakByY)/sys.gamma, ...
            max(gyInitialMergedSlewPeakByY)/sys.gamma);
    elseif isUseSeparateInitialSineFC
        fprintf(['  separate zero-M0 y initial FC target max-|M1|: iy=%d, LIN label=%d, ', ...
                 'target=%.9g 1/m*s, duration=%.6f ms, Gpk=%.6f kHz/m, ', ...
                 'slew=%.6f T/m/s equiv\n'], ...
            gyInitialFCMaxIdx, gyInitialFCMaxIdx-1, gyInitialFCTargetM1Max, ...
            prepModuleDurTarget(1)*1e3, gySineFCGpeak(1)*1e-3, gySineFCSlewPeak(1)/sys.gamma);
    end
end

if isUseInitialZFC
    fprintf('\nInitial z/PAR + cosine merged FC Step 9c summary:\n');
    fprintf('  active PAR-z merged prephaser FC=%d, active cosine initial FC=%d\n', ...
        isUseMergedInitialZPreFC, isUseInitialCosineFC);
    fprintf('  merged cosine M0 contribution = %.9g 1/m\n', gzInitialMergedCosM0);
    fprintf('  actual cosine head M0 through TE1 = %.9g 1/m\n', gzInitialCosHeadM0);
    fprintf('  cosine prep + head M0 closure error = %.9g 1/m\n', gzInitialCosM0ClosureError);
    fprintf('  actual cosine head M1 about TE1 = %.9g 1/m*s\n', gzInitialCosHeadM1AboutTE1);
    fprintf('  actual cosine head M1 about readout start = %.9g 1/m*s\n', ...
        gzInitialCosHeadM1AboutReadoutStart);
    fprintf('  robust merged cosine helper-ref target M1 = %.9g 1/m*s\n', ...
        gzInitialMergedCosTargetM1);
    fprintf('  merged z prephaser M0 range = [%.9g, %.9g] 1/m\n', ...
        min(gzInitialMergedPreM0ByZ), max(gzInitialMergedPreM0ByZ));
    fprintf('  target M1 about TE1 range = [%.9g, %.9g] 1/m*s\n', ...
        min(gzInitialMergedTargetM1AboutTE1ByZ), max(gzInitialMergedTargetM1AboutTE1ByZ));
    fprintf('  helper-ref target M1 range = [%.9g, %.9g] 1/m*s\n', ...
        min(gzInitialMergedTargetM1ByZ), max(gzInitialMergedTargetM1ByZ));
    fprintf('  final M0 range = [%.9g, %.9g] 1/m; final helper-ref M1 range = [%.9g, %.9g] 1/m*s\n', ...
        min(gzInitialMergedFinalM0ByZ), max(gzInitialMergedFinalM0ByZ), ...
        min(gzInitialMergedFinalM1ByZ), max(gzInitialMergedFinalM1ByZ));
    fprintf('  final lobe M1 about TE1 range = [%.9g, %.9g] 1/m*s\n', ...
        min(gzInitialMergedFinalM1AboutTE1ByZ), max(gzInitialMergedFinalM1AboutTE1ByZ));
    fprintf('  predicted lobe + cosine-head M1 about TE1 range = [%.9g, %.9g] 1/m*s\n', ...
        min(gzInitialMergedPredictedTotalM1AtTE1ByZ), ...
        max(gzInitialMergedPredictedTotalM1AtTE1ByZ));
    fprintf(['  worst helper-ref |target M1|: iz=%d, PAR label=%d, target=%.9g 1/m*s, ', ...
             'duration=%.6f ms, Gpk range=[%.6f %.6f] kHz/m, slew range=[%.6f %.6f] T/m/s equiv\n'], ...
        gzInitialFCMaxIdx, gzInitialFCMaxIdx-1, gzInitialFCTargetM1Max, ...
        prepModuleDurTarget(1)*1e3, min(gzInitialMergedGradPeakByZ)*1e-3, ...
        max(gzInitialMergedGradPeakByZ)*1e-3, min(gzInitialMergedSlewPeakByZ)/sys.gamma, ...
        max(gzInitialMergedSlewPeakByZ)/sys.gamma);
end

if isUseAnyInitialXCosFC
    fprintf('\nInitial readout/cosine M0-preserving M1-overlap FC summary:\n');
    fprintf('  common initial prep duration = %.6f ms (grew by %d x 4-raster steps)\n', ...
        prepModuleDurTarget(1)*1e3, initialXCosFCGrowIters);
    if isUseInitialReadoutFC
        fprintf(['  x/readout: M0 target/actual = %.9g / %.9g 1/m, ', ...
                 'M1 target/base/add/final = %.9g / %.9g / %.9g / %.9g 1/m*s, ', ...
                 'H=%.6f kHz/m, B=%.6f kHz/m, Gpk=%.6f kHz/m, slew=%.6f T/m/s equiv\n'], ...
            initialReadoutFCInfo.areaTarget, initialReadoutFCInfo.finalM0, ...
            initialReadoutFCInfo.targetM1, initialReadoutFCInfo.baseM1, ...
            initialReadoutFCInfo.bipM1, initialReadoutFCInfo.finalM1, ...
            initialReadoutFCInfo.H*1e-3, initialReadoutFCInfo.B*1e-3, ...
            initialReadoutFCInfo.gradPeak*1e-3, initialReadoutFCInfo.slewPeak/sys.gamma);
    end
    if isUseSeparateInitialCosineFC
        fprintf(['  z/cosine separate pre: M0 target/actual = %.9g / %.9g 1/m, ' ...
                 'M1 target/base/add/final = %.9g / %.9g / %.9g / %.9g 1/m*s, ', ...
                 'H=%.6f kHz/m, B=%.6f kHz/m, Gpk=%.6f kHz/m, slew=%.6f T/m/s equiv\n'], ...
            initialCosineFCInfo.areaTarget, initialCosineFCInfo.finalM0, ...
            initialCosineFCInfo.targetM1, initialCosineFCInfo.baseM1, ...
            initialCosineFCInfo.bipM1, initialCosineFCInfo.finalM1, ...
            initialCosineFCInfo.H*1e-3, initialCosineFCInfo.B*1e-3, ...
            initialCosineFCInfo.gradPeak*1e-3, initialCosineFCInfo.slewPeak/sys.gamma);
    end
end

% Use the realized Pulseq object areas, not analytic/manual area accounting.
% This includes rasterization, endpoint samples, pre-blip shape, and the
% slew-safe post-ramp exactly as written into the sequence.
waveCarryY = sum(waveAreaY);
waveCarryZ = sum(waveAreaZ);

fprintf('\nWave carry-area summary after %d echoes:\n', Nechoes);
fprintf('  y/sine total area   = %.9g 1/m\n', waveCarryY);
fprintf('  z/cosine total area = %.9g 1/m\n', waveCarryZ);
if isUseWave_sin
    fprintf('  y/sine readout-wave M1 only, echo 1 = %.9g 1/m*s\n', sineReadM1(1));
    fprintf('  y/sine readout-wave M1 only, all echoes = %s 1/m*s\n', mat2str(sineReadM1, 9));
end
if isUseWave_cos
    fprintf('  z/cos ramp-up M1 diagnostic only, echo 1 = %.9g 1/m*s\n', cosRampUpM1(1));
    fprintf('  z/cos ramp-up M1 diagnostic only, all echoes = %s 1/m*s\n', mat2str(cosRampUpM1, 9));
    fprintf('  actual initial cosine-head M0 through TE1 = %.9g 1/m\n', gzInitialCosHeadM0);
    fprintf('  actual initial cosine-head M1 about TE1 = %.9g 1/m*s\n', ...
        gzInitialCosHeadM1AboutTE1);
    fprintf('  robust initial cosine-FC helper-ref target M1 = %.9g 1/m*s\n', ...
        cosInitialFCTargetM1);
end
fprintf('  x-readout module duration      = %.6f ms\n', readoutModuleDurX*1e3);
fprintf('  cosine-derived wave envelope   = %.6f ms\n', waveBlockEnvelopeDur*1e3);

% Worst-case rewinders now include the accumulated wave area in that axis.
gyPostMaxArea = max(abs(-phaseAreasY(:) - waveCarryY));
gzPostMaxArea = max(abs(-phaseAreasZ(:) - waveCarryZ));
gyPostMax = mr.makeTrapezoid('y', 'Area', gyPostMaxArea, ...
    'Duration', TPre, 'system', sys_lowPNS);
gzPostMax = mr.makeTrapezoid('z', 'Area', gzPostMaxArea, ...
    'Duration', TPre, 'system', sys_lowPNS);

%% PE table: accelerated GRE image only

imageYIdx = makeAcceleratedIndexList(Ny, Ry);
imageZIdx = makeAcceleratedIndexList(Nz, Rz);
nImagePE = numel(imageYIdx) * numel(imageZIdx);

fprintf('\nGRE PE table:\n');
fprintf('  Image: %d y/LIN x %d z/PAR = %d PE positions\n', ...
    numel(imageYIdx), numel(imageZIdx), nImagePE);
fprintf('  Readouts: %d PE positions x %d echoes x %d averages = %d\n', ...
    nImagePE, Nechoes, naverage, nImagePE*Nechoes*naverage);

%% Timing calculation

% x prep/dephaser module: echo 1 uses the readout dephaser, later echoes use
% the flyback. gxPrep and optional gzCosPre were already rebuilt above with
% the minimum common duration per echo.
prepModuleDur = zeros(1, Nechoes);
for c = 1:Nechoes
    durs = mr.calcDuration(gxPrep{c});
    if c == 1 && isUseInitialZFC && ~isempty(gzInitialFCByZ)
        gzInitialDurTmp = 0;
        for izTmp = 1:numel(gzInitialFCByZ)
            if ~isempty(gzInitialFCByZ{izTmp})
                gzInitialDurTmp = max(gzInitialDurTmp, mr.calcDuration(gzInitialFCByZ{izTmp}));
            end
        end
        durs(end+1) = gzInitialDurTmp; %#ok<SAGROW>
    elseif c > 1 && isUseInterEchoZFC && ~isempty(gzInterEchoFCByZ)
        gzInterDurTmp = 0;
        for izTmp = 1:size(gzInterEchoFCByZ, 2)
            if ~isempty(gzInterEchoFCByZ{c, izTmp})
                gzInterDurTmp = max(gzInterDurTmp, mr.calcDuration(gzInterEchoFCByZ{c, izTmp}));
            end
        end
        durs(end+1) = gzInterDurTmp; %#ok<SAGROW>
    elseif isUseWave_cos
        durs(end+1) = mr.calcDuration(gzCosPre{c}); %#ok<SAGROW>
    end
    if c == 1 && isUseInitialYFC && ~isempty(gyInitialFCByY)
        gyInitialDurTmp = 0;
        for iyTmp = 1:numel(gyInitialFCByY)
            if ~isempty(gyInitialFCByY{iyTmp})
                gyInitialDurTmp = max(gyInitialDurTmp, mr.calcDuration(gyInitialFCByY{iyTmp}));
            end
        end
        durs(end+1) = gyInitialDurTmp; %#ok<SAGROW>
    elseif c > 1 && isUseInterEchoPEYFC && ~isempty(gyInterEchoFCByY)
        gyInterEchoDurTmp = 0;
        for iyTmp = 1:size(gyInterEchoFCByY, 2)
            if ~isempty(gyInterEchoFCByY{c, iyTmp})
                gyInterEchoDurTmp = max(gyInterEchoDurTmp, mr.calcDuration(gyInterEchoFCByY{c, iyTmp}));
            end
        end
        durs(end+1) = gyInterEchoDurTmp; %#ok<SAGROW>
    elseif isUseAnySineFC && ~isempty(gySineFC{c})
        durs(end+1) = mr.calcDuration(gySineFC{c}); %#ok<SAGROW>
    end
    prepModuleDur(c) = max(durs);
end

rfBlockDur = mr.calcDuration(rf, gz_ss);
rfCenterToEndExc = rfBlockDur - (mr.calcRfCenter(rf) + rf.delay);
t_adc_center = adc.delay + (adc.numSamples / 2) * adc.dwell;

readoutBlockDur = zeros(1, Nechoes);
for c = 1:Nechoes
    durs = [readoutModuleDurX, mr.calcDuration(adc)];
    if isUseWave_sin
        durs(end+1) = mr.calcDuration(gyWave{c}); %#ok<SAGROW>
    end
    if isUseWave_cos
        durs(end+1) = mr.calcDuration(gzWave{c}); %#ok<SAGROW>
    end
    readoutBlockDur(c) = max(durs);
end

% This is only the ordinary PE/slab prephaser block before echo 1. The x
% dephaser/flyback and cosine prewinder are accounted separately in
% prepModuleDur.
prephBlockDurs = mr.calcDuration(gzPreMax);
if ~isUseMergedInitialYPreFC
    prephBlockDurs(end+1) = mr.calcDuration(gyPreMax); %#ok<SAGROW>
end
prephBlockMinDur = max(prephBlockDurs);
spoilerBlockMinDur = mr.calcDuration(gxSpoil, gyPostMax, gzPostMax);

% Echo-to-echo timing. Echo 1 keeps the original split-block timing.
% For centered inter-echo FC, echoes 2+ center the common prep/FC module
% in the gap between neighboring readout gradients.
isUseInterEchoFCStep1 = isUseFlowComp && isFCInterEchoReadoutCos && (Nechoes > 1);
isUseInterEchoCenteredFC = isUseInterEchoFCStep1 || isUseInterEchoYFC || isUseInterEchoZFC;

delayTE = zeros(size(TE));
prepToReadoutGapEcho = prepToReadoutGap * ones(size(TE));

% x readout-gradient start/end inside the readout/wave block. These are
% gradient-object times, not ADC times. They are used only for FC centering.
tReadGradStart = tReadGradStartForFC;
tReadGradEnd   = tReadGradEndForFC;

delayTE(1) = round((TE(1) - rfCenterToEndExc ...
    - prepModuleDur(1) - prepToReadoutGapEcho(1) - t_adc_center) / seq.gradRasterTime) * seq.gradRasterTime;

fcReadoutGradGap = zeros(size(TE));
fcGapBeforeFromReadoutEnd = zeros(size(TE));
fcGapAfterToReadoutStart = zeros(size(TE));
fcCenterError = zeros(size(TE));

for c = 2:Nechoes
    prevReadoutTail = readoutBlockDur(c-1) - t_adc_center;

    if isUseInterEchoCenteredFC
        % Gap between the previous x-readout gradient end and the next
        % x-readout gradient start, measured from the actual gradient
        % support rather than from the surrounding Pulseq block envelope.
        fcReadoutGradGap(c) = TE(c) - TE(c-1) - (tReadGradEnd - tReadGradStart);
        fcFreeGap = fcReadoutGradGap(c) - prepModuleDur(c);
        fcFreeGap = round(fcFreeGap / seq.gradRasterTime) * seq.gradRasterTime;

        if fcFreeGap < -seq.gradRasterTime/10
            error(['Inter-echo centered FC is infeasible for echo %d: ', ...
                   'readout-gradient gap %.6f ms is shorter than prep module %.6f ms.'], ...
                   c, fcReadoutGradGap(c)*1e3, prepModuleDur(c)*1e3);
        end
        fcFreeGap = max(0, fcFreeGap);

        % Split the remaining readout-gradient gap as evenly as the gradient
        % raster allows. A one-raster imbalance may remain when the available
        % gap contains an odd number of raster samples.
        fcGapBeforeFromReadoutEnd(c) = floor((fcFreeGap/2) / seq.gradRasterTime) * seq.gradRasterTime;
        fcGapAfterToReadoutStart(c) = fcFreeGap - fcGapBeforeFromReadoutEnd(c);

        % Convert gradient-referenced gaps to sequence-block delays.
        % The current block loop starts the delay after the previous readout
        % block envelope ends and starts the next readout gradient after
        % gxRead.delay within the next readout/wave block.
        delayTE(c) = fcGapBeforeFromReadoutEnd(c) - (readoutBlockDur(c-1) - tReadGradEnd);
        prepToReadoutGapEcho(c) = fcGapAfterToReadoutStart(c) - tReadGradStart;

        delayTE(c) = round(delayTE(c) / seq.gradRasterTime) * seq.gradRasterTime;
        prepToReadoutGapEcho(c) = round(prepToReadoutGapEcho(c) / seq.gradRasterTime) * seq.gradRasterTime;

        % Module center error relative to the center of the x-readout-gradient gap.
        fcCenterError(c) = 0.5 * (fcGapBeforeFromReadoutEnd(c) - fcGapAfterToReadoutStart(c));
    else
        delayTE(c) = round((TE(c) - TE(c-1) ...
            - prevReadoutTail ...
            - prepModuleDur(c) - prepToReadoutGapEcho(c) - t_adc_center) / seq.gradRasterTime) * seq.gradRasterTime;
    end
end

assert(delayTE(1) >= -seq.gradRasterTime/10, ...
    'TE(1) is too short for RF, echo-1 prep, and readout timing.');
assert(all(delayTE(2:end) >= -seq.gradRasterTime/10), ...
    'Echo spacing is too short. Increase TE spacing or shorten readout/flyback/wave duration.');
assert(all(prepToReadoutGapEcho >= -seq.gradRasterTime/10), ...
    'Inter-echo FC placement produced a negative prep-to-readout block gap. Increase TE spacing or shorten the readout/wave block.');
delayTE = max(delayTE, 0);
prepToReadoutGapEcho = max(prepToReadoutGapEcho, 0);

readoutTrainTime = delayTE(1) + prepModuleDur(1) + prepToReadoutGapEcho(1) + readoutBlockDur(1);
for c = 2:Nechoes
    readoutTrainTime = readoutTrainTime + delayTE(c) + prepModuleDur(c) + prepToReadoutGapEcho(c) + readoutBlockDur(c);
end
delayTR = round((TR - rfBlockDur - readoutTrainTime) / seq.gradRasterTime) * seq.gradRasterTime;

TE_actual = zeros(size(TE));
TE_actual(1) = rfCenterToEndExc + delayTE(1) + prepModuleDur(1) + prepToReadoutGapEcho(1) + t_adc_center;
for c = 2:Nechoes
    TE_actual(c) = TE_actual(c-1) ...
        + (readoutBlockDur(c-1) - t_adc_center) ...
        + delayTE(c) + prepModuleDur(c) + prepToReadoutGapEcho(c) + t_adc_center;
end
TR_actual = rfBlockDur + readoutTrainTime + max(delayTR, spoilerBlockMinDur);
TR_min_feasible = rfBlockDur + readoutTrainTime + spoilerBlockMinDur;

fprintf('\nTiming summary before sequence loop:\n');
fprintf('  rfBlockDur=%.6f ms, rfCenterToEndExc=%.6f ms\n', rfBlockDur*1e3, rfCenterToEndExc*1e3);
fprintf('  PE/slab prephBlockMinDur=%.6f ms, spoilerBlockMinDur=%.6f ms\n', ...
    prephBlockMinDur*1e3, spoilerBlockMinDur*1e3);
fprintf('  prepToReadoutGap input=%.6f ms, inter-echo centered FC=%d, sine FC initial step 4=%d, sine FC inter-echo step 3=%d\n', ...
    prepToReadoutGap*1e3, isUseInterEchoCenteredFC, isUseInitialSineFC, isUseInterEchoYFC);
fprintf('  readout gradient start/end in module = %.6f / %.6f ms\n', ...
    tReadGradStart*1e3, tReadGradEnd*1e3);
fprintf('  adc.delay=%.6f ms, adc center in readout module=%.6f ms\n', adc.delay*1e3, t_adc_center*1e3);
fprintf('  x readout module duration = %.6f ms; requested inter-echo spacings = %s ms\n', ...
    readoutModuleDurX*1e3, mat2str(diff(TE)*1e3));
for c = 1:Nechoes
    fprintf('  Echo %d: prepModuleDur=%.6f ms, readoutBlockDur=%.6f ms, delayTE=%.6f ms, postPrepGap=%.6f ms, TE_target=%.6f ms, TE_actual=%.6f ms, err=%.3f us\n', ...
        c, prepModuleDur(c)*1e3, readoutBlockDur(c)*1e3, delayTE(c)*1e3, prepToReadoutGapEcho(c)*1e3, TE(c)*1e3, TE_actual(c)*1e3, (TE_actual(c)-TE(c))*1e6);
end
if isUseInterEchoCenteredFC
    fprintf('\nInter-echo centered FC placement relative to x readout-gradient support:\n');
    for c = 2:Nechoes
        fprintf(['  Echo %d: ESP=%.6f ms, readoutGradGap=%.6f ms, prepDur=%.6f ms, ', ...
                 'gapBefore=%.6f ms, gapAfter=%.6f ms, centerErr=%.3f us, ', ...
                 'blockDelayBefore=%.6f ms, blockDelayAfter=%.6f ms\n'], ...
            c, (TE(c)-TE(c-1))*1e3, fcReadoutGradGap(c)*1e3, prepModuleDur(c)*1e3, ...
            fcGapBeforeFromReadoutEnd(c)*1e3, fcGapAfterToReadoutStart(c)*1e3, fcCenterError(c)*1e6, ...
            delayTE(c)*1e3, prepToReadoutGapEcho(c)*1e3);
    end
end
fprintf('  TR_target=%.6f ms, delayTR=%.6f ms, TR_actual=%.6f ms, TR_min_feasible=%.6f ms, err=%.3f us\n\n', ...
    TR*1e3, delayTR*1e3, TR_actual*1e3, TR_min_feasible*1e3, (TR_actual-TR)*1e6);

assert(delayTE(1) >= prephBlockMinDur, ...
    'TE(1)=%.6f ms is too short. Minimum feasible TE(1) is %.6f ms with current settings.', ...
    TE(1)*1e3, (rfCenterToEndExc + prephBlockMinDur + prepModuleDur(1) + prepToReadoutGapEcho(1) + t_adc_center)*1e3);
assert(delayTR >= spoilerBlockMinDur, ...
    'TR=%.6f ms is too short. Increase TR to at least %.6f ms.', TR*1e3, TR_min_feasible*1e3);

%% Precompute PE prewinders/rewinders and labels

gyPre = cell(1, Ny);
gyPost = cell(1, Ny);
% If Step 9 merged PE-y FC is active, ordinary gyPre is kept only for
% fallback/diagnostics and may require a duration longer than the z-only
% post-RF prephaser block duration.
gyPreDurForObj = max(gPEPreDur, ceil(mr.calcDuration(gyPreMaxNatural) / sys.gradRasterTime) * sys.gradRasterTime);
for iy = 1:Ny
    gyPre{iy} = mr.makeTrapezoid('y', 'Area', phaseAreasY(iy), ...
        'Duration', gyPreDurForObj, 'system', sys_lowPNS);
    gyPost{iy} = mr.makeTrapezoid('y', 'Area', -phaseAreasY(iy) - waveCarryY, ...
        'Duration', TPre, 'system', sys_lowPNS);
    gyPre{iy}.id = seq.registerGradEvent(gyPre{iy});
    gyPost{iy}.id = seq.registerGradEvent(gyPost{iy});
end

if isUseSlabRephFC
    [gzSlabReph, gzSlabFCInfo] = makeM0M1LobeRefStart( ...
        'z', gzSlabRephM0Target, gzSlabRephM1TargetAtStart, gPEPreDur, sys_lowPNS);
else
    gzSlabReph = mr.makeTrapezoid('z', 'Area', gzSlabRephM0Target, ...
        'Duration', gPEPreDur, 'system', sys_lowPNS);
    gzSlabFCInfo = struct('active', false, 'T', gPEPreDur, ...
        'areaTarget', gzSlabRephM0Target, 'targetM1', gzSlabRephM1TargetAtStart);
end
gzSlabReph.id = seq.registerGradEvent(gzSlabReph);

gzPreComb = cell(1, Nz);    % legacy/fallback: slab rephaser + z/PAR prephaser
gzParPre = cell(1, Nz);     % separated z/PAR prephaser only, used for diagnostics/fallback clarity
gzPost = cell(1, Nz);
for iz = 1:Nz
    gzParPreNatural = mr.makeTrapezoid('z', 'Area', phaseAreasZ(iz), ...
        'system', sys_lowPNS);
    gzParPreDurForObj = max(gPEPreDur, ...
        ceil(mr.calcDuration(gzParPreNatural) / sys.gradRasterTime) * sys.gradRasterTime);
    gzParPre{iz} = mr.makeTrapezoid('z', 'Area', phaseAreasZ(iz), ...
        'Duration', gzParPreDurForObj, 'system', sys_lowPNS);

    if ~isUseMergedInitialZPreFC
        gzPreCombNatural = mr.makeTrapezoid('z', 'Area', phaseAreasZ(iz) + gzSlabRephM0Target, ...
            'system', sys_lowPNS);
        gzPreCombDurForObj = max(gPEPreDur, ...
            ceil(mr.calcDuration(gzPreCombNatural) / sys.gradRasterTime) * sys.gradRasterTime);
        gzPreComb{iz} = mr.makeTrapezoid('z', 'Area', phaseAreasZ(iz) + gzSlabRephM0Target, ...
            'Duration', gzPreCombDurForObj, 'system', sys_lowPNS);
    else
        gzPreComb{iz} = [];
    end

    gzPost{iz} = mr.makeTrapezoid('z', 'Area', -phaseAreasZ(iz) - waveCarryZ, ...
        'Duration', TPre, 'system', sys_lowPNS);
    gzParPre{iz}.id = seq.registerGradEvent(gzParPre{iz});
    if ~isempty(gzPreComb{iz})
        gzPreComb{iz}.id = seq.registerGradEvent(gzPreComb{iz});
    end
    gzPost{iz}.id = seq.registerGradEvent(gzPost{iz});
end

for c = 1:Nechoes
    gxPrep{c}.id = seq.registerGradEvent(gxPrep{c});
    if isUseWave_cos && ~(c == 1 && isUseMergedInitialZPreFC) && ~(c > 1 && isUseInterEchoZFC)
        gzCosPre{c}.id = seq.registerGradEvent(gzCosPre{c});
    end
    if c == 1 && isUseInitialYFC
        % Echo-1 y FC is PE-index dependent; register the per-y objects below.
    elseif c > 1 && isUseInterEchoPEYFC
        % Inter-echo y FC is PE-index dependent; register below.
    elseif isUseAnySineFC && ~isempty(gySineFC{c})
        gySineFC{c}.id = seq.registerGradEvent(gySineFC{c});
    end
end
if isUseInitialYFC
    for iyPEFC = 1:Ny
        gyInitialFCByY{iyPEFC}.id = seq.registerGradEvent(gyInitialFCByY{iyPEFC});
    end
end
if isUseInterEchoPEYFC
    for c = 2:Nechoes
        for iyPEFC = 1:Ny
            if ~isempty(gyInterEchoFCByY{c, iyPEFC})
                gyInterEchoFCByY{c, iyPEFC}.id = seq.registerGradEvent(gyInterEchoFCByY{c, iyPEFC});
            end
        end
    end
end
if isUseInitialZFC
    for izPARFC = 1:Nz
        gzInitialFCByZ{izPARFC}.id = seq.registerGradEvent(gzInitialFCByZ{izPARFC});
    end
end
if isUseInterEchoZFC
    for c = 2:Nechoes
        for izPARFC = 1:Nz
            if ~isempty(gzInterEchoFCByZ{c, izPARFC})
                gzInterEchoFCByZ{c, izPARFC}.id = seq.registerGradEvent(gzInterEchoFCByZ{c, izPARFC});
            end
        end
    end
end
gxRead.id = seq.registerGradEvent(gxRead);
gxSpoil.id = seq.registerGradEvent(gxSpoil);
[~, rf.shapeIDs] = seq.registerRfEvent(rf);

% GRE image labels use global 0-based reconstructed matrix coordinates.
lblLIN_img = cell(1, Ny);
for iy = 1:Ny
    lblLIN_img{iy} = mr.makeLabel('SET', 'LIN', iy - 1);
end
lblPAR_img = cell(1, Nz);
for iz = 1:Nz
    lblPAR_img{iz} = mr.makeLabel('SET', 'PAR', iz - 1);
end
lblECO = cell(1, Nechoes);
for c = 1:Nechoes
    lblECO{c} = mr.makeLabel('SET', 'ECO', c - 1);
end
lblAVG = cell(1, naverage);
for iAvg = 1:naverage
    lblAVG{iAvg} = mr.makeLabel('SET', 'AVG', iAvg - 1);
end
lblSET_img = mr.makeLabel('SET', 'SET', 0);
lblRefOff  = mr.makeLabel('SET', 'REF', false);
lblImaOff  = mr.makeLabel('SET', 'IMA', false);
useIceRefscanLabels = true;

%% Sequence loop

% RF spoiling counters are continuous through GRE dummy and image blocks.
rf_phase = 0;
rf_inc = 0;

% Dummy scans use center PE timing but no ADC/labels.
yCenterIdx = floor(Ny/2) + 1;
zCenterIdx = floor(Nz/2) + 1;

fprintf('\nBuilding sequence blocks...\n');
tic;

for iDummy = 1:NdummyGre
    [rf, adc, rf_phase, rf_inc] = updateRfAndAdcPhase(rf, adc, rf_phase, rf_inc, rfSpoilingInc);
    seq.addBlock(rf, gz_ss);
    if isUseMergedInitialYPreFC && isUseMergedInitialZPreFC
        seq.addBlock(mr.makeDelay(delayTE(1)), gzSlabReph);
    elseif isUseMergedInitialYPreFC
        seq.addBlock(mr.makeDelay(delayTE(1)), gzPreComb{zCenterIdx});
    elseif isUseMergedInitialZPreFC
        seq.addBlock(mr.makeDelay(delayTE(1)), gyPre{yCenterIdx}, gzSlabReph);
    else
        seq.addBlock(mr.makeDelay(delayTE(1)), gyPre{yCenterIdx}, gzPreComb{zCenterIdx});
    end
    for c = 1:Nechoes
        if c > 1
            seq.addBlock(mr.makeDelay(delayTE(c)));
        end
        addPrepModule(seq, gxPrep{c}, gzCosPre, gySineFC, c, isUseWave_cos, isUseAnySineFC, gyInitialFCByY, yCenterIdx, isUseInitialYFC, gzInitialFCByZ, zCenterIdx, isUseInitialZFC, gyInterEchoFCByY, isUseInterEchoPEYFC, gzInterEchoFCByZ, isUseInterEchoZFC);
        addModuleGap(seq, prepToReadoutGapEcho(c));
        addReadoutBlockNoAdc(seq, gxRead, gyWave, gzWave, c, isUseWave_sin, isUseWave_cos);
    end
    seq.addBlock(mr.makeDelay(delayTR), gxSpoil, gyPost{yCenterIdx}, gzPost{zCenterIdx});
end

nImageReadouts = 0;
for iAvg = 1:naverage
    for iy = imageYIdx
        for iz = imageZIdx
            [rf, adc, rf_phase, rf_inc] = updateRfAndAdcPhase(rf, adc, rf_phase, rf_inc, rfSpoilingInc);
            seq.addBlock(rf, gz_ss);
            if isUseMergedInitialYPreFC && isUseMergedInitialZPreFC
                seq.addBlock(mr.makeDelay(delayTE(1)), gzSlabReph);
            elseif isUseMergedInitialYPreFC
                seq.addBlock(mr.makeDelay(delayTE(1)), gzPreComb{iz});
            elseif isUseMergedInitialZPreFC
                seq.addBlock(mr.makeDelay(delayTE(1)), gyPre{iy}, gzSlabReph);
            else
                seq.addBlock(mr.makeDelay(delayTE(1)), gyPre{iy}, gzPreComb{iz});
            end
            for c = 1:Nechoes
                if c > 1
                    seq.addBlock(mr.makeDelay(delayTE(c)));
                end
                addPrepModule(seq, gxPrep{c}, gzCosPre, gySineFC, c, isUseWave_cos, isUseAnySineFC, gyInitialFCByY, iy, isUseInitialYFC, gzInitialFCByZ, iz, isUseInitialZFC, gyInterEchoFCByY, isUseInterEchoPEYFC, gzInterEchoFCByZ, isUseInterEchoZFC);
                addModuleGap(seq, prepToReadoutGapEcho(c));
                addReadoutBlockWithAdc(seq, gxRead, gyWave, gzWave, adc, c, ...
                    isUseWave_sin, isUseWave_cos, lblLIN_img{iy}, lblPAR_img{iz}, lblECO{c}, lblAVG{iAvg}, lblSET_img, lblImaOff, lblRefOff, useIceRefscanLabels);
                nImageReadouts = nImageReadouts + 1;
            end
            seq.addBlock(mr.makeDelay(delayTR), gxSpoil, gyPost{iy}, gzPost{iz});
        end
    end
end

fprintf('GRE sequence block generation took %.3f seconds.\n', toc);
fprintf('GRE acquired image readouts: %d\n', nImageReadouts);
fprintf('GRE RF excitations including dummy: %d\n', ...
    NdummyGre + nImagePE*naverage);

%% Build and append slab-selective FLASH calibration

fprintf('\nPreparing appended FLASH calibration acquisition...\n');

MODE_NOWAVE = 1;
MODE_SIN    = 2;
MODE_COS    = 3;
calibModeNames = {'nowave', 'sin', 'cos'};

% Calibration uses the shared TRA geometry and readout settings, but keeps a
% separate readout spoiler value. The RF pulse is slab-selective and matches
% the GRE excitation geometry.
[rfCal, gzCalSs, gzCalSsReph] = mr.makeSincPulse(alpha*pi/180, sys_lowPNS, ...
    'Duration', rfDuration, ...
    'SliceThickness', slabExciteThickness, ...
    'apodization', rfApodization, ...
    'timeBwProduct', rfTBP, ...
    'use', 'excitation');

calibDwell = dwell;
calibTread = calibDwell * Nx_os;
groCal = mr.makeTrapezoid(ax.d1, ...
    'Amplitude', Nx*deltak(ax.n1)/calibTread, ...
    'FlatTime', ceil((calibTread + sys.adcDeadTime) ...
        / sys.gradRasterTime) * sys.gradRasterTime, ...
    'system', sys);
adcCal = mr.makeAdc(Nx_os, 'Duration', calibTread, ...
    'Delay', groCal.riseTime, 'system', sys);
assert(adcCal.numSamples == Nx_os, 'Calibration ADC sample count mismatch.');

groCalPre = mr.makeTrapezoid(ax.d1, ...
    'Area', -groCal.amplitude * ...
        (adcCal.dwell*(adcCal.numSamples/2+0.5) + 0.5*groCal.riseTime), ...
    'system', sys_lowPNS);

% Match the GRE physical PE convention exactly: increasing MATLAB/TWIX
% indices correspond to increasing k-space coordinates, from negative to
% positive. These are the desired PE moments at the ADC center for the
% no-wave baseline of each calibration line.
calibParAreas = ((0:N(ax.n2)-1) - N(ax.n2)/2) * deltak(ax.n2);
calibLinAreas = ((0:N(ax.n3)-1) - N(ax.n3)/2) * deltak(ax.n3);
calibParMaxAbsArea = max(abs(calibParAreas));
calibLinMaxAbsArea = max(abs(calibLinAreas));

assert(all(diff(calibParAreas) > 0), ...
    'Calibration PAR target areas must increase from negative to positive.');
assert(all(diff(calibLinAreas) > 0), ...
    'Calibration LIN target areas must increase from negative to positive.');
assert(abs(calibParAreas(floor(N(ax.n2)/2)+1)) < 1e-12, ...
    'Calibration PAR center index must have zero PE area.');
assert(abs(calibLinAreas(floor(N(ax.n3)/2)+1)) < 1e-12, ...
    'Calibration LIN center index must have zero PE area.');

gpeCalParMax = mr.makeTrapezoid(ax.d2, ...
    'Area', calibParMaxAbsArea, 'system', sys_lowPNS);
gpeCalLinMax = mr.makeTrapezoid(ax.d3, ...
    'Area', calibLinMaxAbsArea, 'system', sys_lowPNS);

[groCalRead, groCalSp] = mr.splitGradientAt( ...
    groCal, groCal.riseTime + groCal.flatTime);
if calib_ro_spoil > 0
    groCalSp = mr.makeExtendedTrapezoidArea(groCal.channel, ...
        groCal.amplitude, 0, ...
        deltak(ax.n1)/2*N(ax.n1)*calib_ro_spoil, sys_lowPNS);
end

calibPreDur = max([mr.calcDuration(groCalPre), ...
    mr.calcDuration(gpeCalParMax), mr.calcDuration(gpeCalLinMax)]);
calibPreDur = ceil(calibPreDur/sys.gradRasterTime)*sys.gradRasterTime;
groCalPre = mr.makeTrapezoid(ax.d1, 'Area', groCalPre.area, ...
    'Duration', calibPreDur, 'system', sys_lowPNS);
gpeCalParPreMax = mr.makeTrapezoid(ax.d2, 'Area', gpeCalParMax.area, ...
    'Duration', calibPreDur, 'system', sys_lowPNS);
gpeCalLinPreMax = mr.makeTrapezoid(ax.d3, 'Area', gpeCalLinMax.area, ...
    'Duration', calibPreDur, 'system', sys_lowPNS);

groCalRead.delay = mr.calcDuration(groCalPre);
adcCal.delay = groCalRead.delay + groCal.riseTime;
groCalRead = mr.addGradients({groCalRead, groCalPre}, 'system', sys);

% Scale positive maximum-area templates to the signed physical target areas.
% The first index is the most negative encoded location, the center index is
% zero, and the final index is the most positive encoded location.
calibParScales = calibParAreas / calibParMaxAbsArea;
calibLinScales = calibLinAreas / calibLinMaxAbsArea;
calibAreaTol = max(1e-9, 1e-10 * max([calibParMaxAbsArea, calibLinMaxAbsArea]));

gpeCalParPre_nowave  = cell(1, N(ax.n2));
gpeCalParPost_nowave = cell(1, N(ax.n2));
gpeCalParPre_cos     = cell(1, N(ax.n2));
gpeCalParPost_cos    = cell(1, N(ax.n2));
gpeCalLinPre_nowave  = cell(1, N(ax.n3));
gpeCalLinPost_nowave = cell(1, N(ax.n3));
gpeCalLinPre_sin     = cell(1, N(ax.n3));
gpeCalLinPost_sin    = cell(1, N(ax.n3));
calibPostDurations = mr.calcDuration(groCalSp);

% TRA mapping: PAR/z carries cosine; LIN/y carries sine.
for izCal = 1:N(ax.n2)
    gpePreNow = mr.scaleGrad(gpeCalParPreMax, calibParScales(izCal));
    gpeCalParPre_nowave{izCal} = gpePreNow;
    gpeCalParPost_nowave{izCal} = mr.scaleGrad( ...
        gpeCalParMax, -calibParScales(izCal));
    assert(abs(gpeCalParPre_nowave{izCal}.area - calibParAreas(izCal)) <= calibAreaTol, ...
        'Calibration PAR prephaser area/order mismatch at index %d.', izCal);
    assert(abs(gpeCalParPost_nowave{izCal}.area + calibParAreas(izCal)) <= calibAreaTol, ...
        'Calibration PAR rewinder area mismatch at index %d.', izCal);

    debugThis = calibWaveDebugFlag && (izCal == 1);
    [gpeCalParPre_cos{izCal}, gpeCalParPost_cos{izCal}] = ...
        defineCosineWaveGradient4Calib(calibTread, sys, sys_lowPNS, ...
            Ncycles, gwave_max, swave_max, gpePreNow, groCal, adcCal, ...
            physical_slew_max, calibWaveInfoFlag && (izCal == 1), debugThis);

    gpeCalParPre_nowave{izCal}.id = seq.registerGradEvent(gpeCalParPre_nowave{izCal});
    gpeCalParPost_nowave{izCal}.id = seq.registerGradEvent(gpeCalParPost_nowave{izCal});
    gpeCalParPre_cos{izCal}.id = seq.registerGradEvent(gpeCalParPre_cos{izCal});
    gpeCalParPost_cos{izCal}.id = seq.registerGradEvent(gpeCalParPost_cos{izCal});
    calibPostDurations(end+1:end+2) = [ ...
        mr.calcDuration(gpeCalParPost_nowave{izCal}), ...
        mr.calcDuration(gpeCalParPost_cos{izCal})]; %#ok<SAGROW>
end

for iyCal = 1:N(ax.n3)
    gpePreNow = mr.scaleGrad(gpeCalLinPreMax, calibLinScales(iyCal));
    gpeCalLinPre_nowave{iyCal} = gpePreNow;
    gpeCalLinPost_nowave{iyCal} = mr.scaleGrad( ...
        gpeCalLinMax, -calibLinScales(iyCal));
    assert(abs(gpeCalLinPre_nowave{iyCal}.area - calibLinAreas(iyCal)) <= calibAreaTol, ...
        'Calibration LIN prephaser area/order mismatch at index %d.', iyCal);
    assert(abs(gpeCalLinPost_nowave{iyCal}.area + calibLinAreas(iyCal)) <= calibAreaTol, ...
        'Calibration LIN rewinder area mismatch at index %d.', iyCal);

    debugThis = calibWaveDebugFlag && (iyCal == 1);
    [gpeCalLinPre_sin{iyCal}, gpeCalLinPost_sin{iyCal}] = ...
        defineSineWaveGradient4Calib(calibTread, sys, sys_lowPNS, ...
            Ncycles, gwave_max, swave_max, gpePreNow, groCal, adcCal, ...
            physical_slew_max, calibWaveInfoFlag && (iyCal == 1), debugThis);

    gpeCalLinPre_nowave{iyCal}.id = seq.registerGradEvent(gpeCalLinPre_nowave{iyCal});
    gpeCalLinPost_nowave{iyCal}.id = seq.registerGradEvent(gpeCalLinPost_nowave{iyCal});
    gpeCalLinPre_sin{iyCal}.id = seq.registerGradEvent(gpeCalLinPre_sin{iyCal});
    gpeCalLinPost_sin{iyCal}.id = seq.registerGradEvent(gpeCalLinPost_sin{iyCal});
    calibPostDurations(end+1:end+2) = [ ...
        mr.calcDuration(gpeCalLinPost_nowave{iyCal}), ...
        mr.calcDuration(gpeCalLinPost_sin{iyCal})]; %#ok<SAGROW>
end

% Modes that do not wave on an axis use the no-wave events for that axis.
gpeCalParPreByMode  = {gpeCalParPre_nowave,  gpeCalParPre_nowave,  gpeCalParPre_cos};
gpeCalParPostByMode = {gpeCalParPost_nowave, gpeCalParPost_nowave, gpeCalParPost_cos};
gpeCalLinPreByMode  = {gpeCalLinPre_nowave,  gpeCalLinPre_sin,     gpeCalLinPre_nowave};
gpeCalLinPostByMode = {gpeCalLinPost_nowave, gpeCalLinPost_sin,    gpeCalLinPost_nowave};

calibPostBlockDur = ceil(max(calibPostDurations)/sys.gradRasterTime) ...
    * sys.gradRasterTime;
calibRfBlockDur = mr.calcDuration(rfCal, gzCalSs);
calibSlabRephDur = mr.calcDuration(gzCalSsReph);
calibReadBlockDur = max([mr.calcDuration(groCalRead), mr.calcDuration(adcCal)]);
calibTR = calibRfBlockDur + calibSlabRephDur + calibReadBlockDur + calibPostBlockDur;
calibTE = calibRfBlockDur - (rfCal.delay + mr.calcRfCenter(rfCal)) ...
    + calibSlabRephDur + adcCal.delay ...
    + adcCal.dwell*(adcCal.numSamples/2+0.5);

fprintf(['Calibration timing: RF block=%.6f ms, standalone slab rephaser=%.6f ms, ', ...
    'readout block=%.6f ms, post block=%.6f ms, TE=%.6f ms, TR=%.6f ms.\n'], ...
    calibRfBlockDur*1e3, calibSlabRephDur*1e3, calibReadBlockDur*1e3, ...
    calibPostBlockDur*1e3, calibTE*1e3, calibTR*1e3);

% Calibration acquisition table with compact local LIN/PAR labels.
kyCal1 = centerBlockIndices(N(ax.n3), Ncalib1);
kyCal2 = centerBlockIndices(N(ax.n3), Ncalib2);
kyCalAcs = centerBlockIndices(N(ax.n3), NacsCal);
kzCal1 = centerBlockIndices(N(ax.n2), Ncalib1);
kzCal2 = centerBlockIndices(N(ax.n2), Ncalib2);
kzCalAcs = centerBlockIndices(N(ax.n2), NacsCal);

% centerBlockIndices returns ascending physical indices. Combined with the
% target-area arrays above, compact local LIN/PAR labels now map directly to
% negative-to-positive physical k-space in every calibration SET.
assert(all(diff(calibLinAreas(kyCal1)) > 0), ...
    'Calibration LIN-wide block is not ordered negative-to-positive.');
assert(all(diff(calibParAreas(kzCal1)) > 0), ...
    'Calibration PAR-wide block is not ordered negative-to-positive.');
assert(all(diff(calibLinAreas(kyCalAcs)) > 0), ...
    'Calibration ACS LIN block is not ordered negative-to-positive.');
assert(all(diff(calibParAreas(kzCalAcs)) > 0), ...
    'Calibration ACS PAR block is not ordered negative-to-positive.');
fprintf(['Calibration PE ordering: negative-to-positive for both LIN/y and ', ...
    'PAR/z; compact local labels increase with physical k-space.\n']);

calParts = struct('id', {}, 'name', {}, 'mode', {}, ...
    'kyList', {}, 'kzList', {}, 'isACS', {});
calParts(1) = struct('id',0,'name','nowave_kywide_kznarrow', ...
    'mode',MODE_NOWAVE,'kyList',kyCal1,'kzList',kzCal2,'isACS',false);
calParts(2) = struct('id',1,'name','sin_kywide_kznarrow', ...
    'mode',MODE_SIN,'kyList',kyCal1,'kzList',kzCal2,'isACS',false);
calParts(3) = struct('id',2,'name','nowave_kzwide_kynarrow', ...
    'mode',MODE_NOWAVE,'kyList',kyCal2,'kzList',kzCal1,'isACS',false);
calParts(4) = struct('id',3,'name','cos_kzwide_kynarrow', ...
    'mode',MODE_COS,'kyList',kyCal2,'kzList',kzCal1,'isACS',false);
calParts(5) = struct('id',4,'name','acs_nowave_center', ...
    'mode',MODE_NOWAVE,'kyList',kyCalAcs,'kzList',kzCalAcs,'isACS',true);

calAcqTable = struct('partArrayIdx', {}, 'partID', {}, 'mode', {}, ...
    'isACS', {}, 'iPhys', {}, 'jPhys', {}, 'iLocal', {}, 'jLocal', {});
calPartStart = zeros(1, numel(calParts));
calPartStop = zeros(1, numel(calParts));
for pCal = 1:numel(calParts)
    calPartStart(pCal) = numel(calAcqTable) + 1;
    for jLocal = 1:numel(calParts(pCal).kyList)
        for iLocal = 1:numel(calParts(pCal).kzList)
            row = struct;
            row.partArrayIdx = pCal;
            row.partID = calParts(pCal).id;
            row.mode = calParts(pCal).mode;
            row.isACS = calParts(pCal).isACS;
            row.iPhys = calParts(pCal).kzList(iLocal);
            row.jPhys = calParts(pCal).kyList(jLocal);
            row.iLocal = iLocal;
            row.jLocal = jLocal;
            calAcqTable(end+1) = row; %#ok<SAGROW>
        end
    end
    calPartStop(pCal) = numel(calAcqTable);
end
nCalReadoutsExpected = 4*Ncalib1*Ncalib2 + NacsCal*NacsCal;
assert(numel(calAcqTable) == nCalReadoutsExpected, ...
    'Calibration acquisition-table length mismatch.');

maxLocalLin = max(arrayfun(@(p) numel(p.kyList), calParts));
maxLocalPar = max(arrayfun(@(p) numel(p.kzList), calParts));
lblLIN_cal = cell(1, maxLocalLin);
lblPAR_cal = cell(1, maxLocalPar);
for ii = 1:maxLocalLin, lblLIN_cal{ii} = mr.makeLabel('SET','LIN',ii-1); end
for ii = 1:maxLocalPar, lblPAR_cal{ii} = mr.makeLabel('SET','PAR',ii-1); end
lblSET_cal = cell(1, numel(calParts));
for pCal = 1:numel(calParts)
    lblSET_cal{pCal} = mr.makeLabel('SET','SET',calParts(pCal).id);
end
lblECO_cal = mr.makeLabel('SET','ECO',0);
lblAVG_cal = mr.makeLabel('SET','AVG',0);
lblRefOn = mr.makeLabel('SET','REF',true);
lblImaOffCal = mr.makeLabel('SET','IMA',false);

% Register invariant calibration objects.
groCalRead.id = seq.registerGradEvent(groCalRead);
groCalSp.id = seq.registerGradEvent(groCalSp);
gzCalSsReph.id = seq.registerGradEvent(gzCalSsReph);
[~, rfCal.shapeIDs] = seq.registerRfEvent(rfCal);

fprintf('Appending FLASH calibration blocks...\n');
rfCalPhase = 0;
rfCalInc = 0;
nCalReadouts = 0;
dummyTableIdx = mod((-NdummyCal:-1), numel(calAcqTable)) + 1;
tic;

for kk = 1:numel(dummyTableIdx)
    row = calAcqTable(dummyTableIdx(kk));
    [rfCal, adcCal, rfCalPhase, rfCalInc] = updateRfAndAdcPhase( ...
        rfCal, adcCal, rfCalPhase, rfCalInc, rfSpoilingInc);
    seq.addBlock(rfCal, gzCalSs);
    seq.addBlock(gzCalSsReph);
    seq.addBlock(groCalRead, ...
        gpeCalParPreByMode{row.mode}{row.iPhys}, ...
        gpeCalLinPreByMode{row.mode}{row.jPhys});
    seq.addBlock(mr.makeDelay(calibPostBlockDur), groCalSp, ...
        gpeCalParPostByMode{row.mode}{row.iPhys}, ...
        gpeCalLinPostByMode{row.mode}{row.jPhys});
end

for pCal = 1:numel(calParts)
    partRows = calAcqTable(calPartStart(pCal):calPartStop(pCal));
    settleIdx = mod((-NsettlePerPart:-1), numel(partRows)) + 1;
    for kk = 1:numel(settleIdx)
        row = partRows(settleIdx(kk));
        [rfCal, adcCal, rfCalPhase, rfCalInc] = updateRfAndAdcPhase( ...
            rfCal, adcCal, rfCalPhase, rfCalInc, rfSpoilingInc);
        seq.addBlock(rfCal, gzCalSs);
        seq.addBlock(gzCalSsReph);
        seq.addBlock(groCalRead, ...
            gpeCalParPreByMode{row.mode}{row.iPhys}, ...
            gpeCalLinPreByMode{row.mode}{row.jPhys});
        seq.addBlock(mr.makeDelay(calibPostBlockDur), groCalSp, ...
            gpeCalParPostByMode{row.mode}{row.iPhys}, ...
            gpeCalLinPostByMode{row.mode}{row.jPhys});
    end

    for kk = calPartStart(pCal):calPartStop(pCal)
        row = calAcqTable(kk);
        [rfCal, adcCal, rfCalPhase, rfCalInc] = updateRfAndAdcPhase( ...
            rfCal, adcCal, rfCalPhase, rfCalInc, rfSpoilingInc);
        seq.addBlock(rfCal, gzCalSs);
        seq.addBlock(gzCalSsReph);
        seq.addBlock(adcCal, groCalRead, ...
            gpeCalParPreByMode{row.mode}{row.iPhys}, ...
            gpeCalLinPreByMode{row.mode}{row.jPhys}, ...
            lblPAR_cal{row.iLocal}, lblLIN_cal{row.jLocal}, ...
            lblSET_cal{pCal}, lblECO_cal, lblAVG_cal, lblRefOn, lblImaOffCal);
        seq.addBlock(mr.makeDelay(calibPostBlockDur), groCalSp, ...
            gpeCalParPostByMode{row.mode}{row.iPhys}, ...
            gpeCalLinPostByMode{row.mode}{row.jPhys});
        nCalReadouts = nCalReadouts + 1;
    end
end

fprintf('Calibration blocks appended in %.3f seconds.\n', toc);
fprintf('Calibration readouts: %d; RF excitations: %d dummy + %d settling + %d acquired.\n', ...
    nCalReadouts, NdummyCal, NsettlePerPart*numel(calParts), nCalReadouts);
assert(nCalReadouts == nCalReadoutsExpected, ...
    'Unexpected calibration readout count.');

%% Combined label and TWIX-routing validation

adc_lbl = seq.evalLabels('evolution','adc');
requiredLabels = {'LIN','PAR','ECO','AVG','SET','REF','IMA'};
for ii = 1:numel(requiredLabels)
    assert(isfield(adc_lbl, requiredLabels{ii}), ...
        'Required ADC label %s was not found.', requiredLabels{ii});
end

expectedImageReadouts = nImagePE * Nechoes * naverage;
assert(nImageReadouts == expectedImageReadouts, ...
    'Unexpected GRE image readout count.');
assert(numel(adc_lbl.LIN) == expectedImageReadouts + nCalReadoutsExpected, ...
    'Unexpected total ADC count.');

% Exact GRE order: AVG -> LIN -> PAR -> ECO.
expectedLIN_img = zeros(expectedImageReadouts,1);
expectedPAR_img = zeros(expectedImageReadouts,1);
expectedECO_img = zeros(expectedImageReadouts,1);
expectedAVG_img = zeros(expectedImageReadouts,1);
kkExpected = 0;
for iAvg = 1:naverage
    for iy = imageYIdx
        for iz = imageZIdx
            for c = 1:Nechoes
                kkExpected = kkExpected + 1;
                expectedLIN_img(kkExpected) = iy-1;
                expectedPAR_img(kkExpected) = iz-1;
                expectedECO_img(kkExpected) = c-1;
                expectedAVG_img(kkExpected) = iAvg-1;
            end
        end
    end
end

imgRange = 1:expectedImageReadouts;
imgLIN = adc_lbl.LIN(imgRange); imgLIN = imgLIN(:);
imgPAR = adc_lbl.PAR(imgRange); imgPAR = imgPAR(:);
imgECO = adc_lbl.ECO(imgRange); imgECO = imgECO(:);
imgAVG = adc_lbl.AVG(imgRange); imgAVG = imgAVG(:);
imgSET = adc_lbl.SET(imgRange); imgSET = imgSET(:);
imgREF = adc_lbl.REF(imgRange); imgREF = imgREF(:);
imgIMA = adc_lbl.IMA(imgRange); imgIMA = imgIMA(:);
assert(all(imgLIN == expectedLIN_img), 'GRE LIN order mismatch.');
assert(all(imgPAR == expectedPAR_img), 'GRE PAR order mismatch.');
assert(all(imgECO == expectedECO_img), 'GRE ECO order mismatch.');
assert(all(imgAVG == expectedAVG_img), 'GRE AVG order mismatch.');
assert(all(imgSET == 0), 'GRE image must use SET=0.');
assert(all(imgREF == 0), 'GRE image must use REF=false.');
assert(all(imgIMA == 0), 'GRE image must use IMA=false.');

calRange = expectedImageReadouts + (1:nCalReadoutsExpected);
calSET = adc_lbl.SET(calRange); calSET = calSET(:);
calPAR = adc_lbl.PAR(calRange); calPAR = calPAR(:);
calLIN = adc_lbl.LIN(calRange); calLIN = calLIN(:);
calECO = adc_lbl.ECO(calRange); calECO = calECO(:);
calAVG = adc_lbl.AVG(calRange); calAVG = calAVG(:);
calREF = adc_lbl.REF(calRange); calREF = calREF(:);
calIMA = adc_lbl.IMA(calRange); calIMA = calIMA(:);
expectedSET_cal = [calAcqTable.partID]';
expectedPAR_cal = [calAcqTable.iLocal]' - 1;
expectedLIN_cal = [calAcqTable.jLocal]' - 1;
assert(all(calSET == expectedSET_cal), 'Calibration SET order mismatch.');
assert(all(calPAR == expectedPAR_cal), 'Calibration compact PAR order mismatch.');
assert(all(calLIN == expectedLIN_cal), 'Calibration compact LIN order mismatch.');
assert(all(calECO == 0), 'Calibration ECO must be zero.');
assert(all(calAVG == 0), 'Calibration AVG must be zero.');
assert(all(calREF ~= 0), 'Every calibration ADC must use REF=true.');
assert(all(calIMA == 0), 'Calibration must use IMA=false.');

calTriples = [calSET, calPAR, calLIN];
assert(size(unique(calTriples,'rows'),1) == nCalReadoutsExpected, ...
    'Duplicate calibration [SET,PAR,LIN] labels found.');
assert(max(calLIN) == Ncalib1-1, ...
    'Calibration refscan LIN extent is not Ncalib1.');
assert(max(calPAR) == Ncalib1-1, ...
    'Calibration refscan PAR extent is not Ncalib1.');
assert(max(calSET) == 4, ...
    'Calibration refscan SET extent is not 0:4.');

acsMask = calSET == 4;
acsPairs = [calPAR(acsMask), calLIN(acsMask)];
[acsParExpected, acsLinExpected] = ndgrid(0:NacsCal-1, 0:NacsCal-1);
acsPairsExpected = [acsParExpected(:), acsLinExpected(:)];
assert(isempty(setdiff(acsPairsExpected,acsPairs,'rows')) && ...
    isempty(setdiff(acsPairs,acsPairsExpected,'rows')), ...
    'Calibration ACS is not stored at local PAR/LIN 0:(NacsCal-1).');

fprintf(['Combined routing validated: GRE image ADCs=%d; calibration refscan ADCs=%d.\n'], ...
    expectedImageReadouts, nCalReadoutsExpected);
fprintf('Calibration logical refscan extent: LIN=%d, PAR=%d, SET=%d.\n', ...
    Ncalib1, Ncalib1, numel(calParts));

%% Timing check
[ok, error_report] = seq.checkTiming;
if ok
    fprintf('Timing check passed successfully.\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end

%% Concise sequence definitions and write
seq.setDefinition('FOV', fov);
seq.setDefinition('TargetFOV', target_fov);
seq.setDefinition('Nx', Nx);
seq.setDefinition('Ny', Ny);
seq.setDefinition('Nz', Nz);
seq.setDefinition('Nx_os', Nx_os);
seq.setDefinition('OrientationMapping', slOrientation);
seq.setDefinition('SliceThickness', slabExciteThickness);
seq.setDefinition('EncodedSlabThickness', slabEncodeThickness);
seq.setDefinition('SliceOversampling', sliceOS);
seq.setDefinition('FlipAngle', alpha);
seq.setDefinition('TE', TE);
seq.setDefinition('TR', TR);
seq.setDefinition('Nechoes', Nechoes);
seq.setDefinition('Averages', naverage);
seq.setDefinition('Ry', Ry);
seq.setDefinition('Rz', Rz);
seq.setDefinition('Ny_meas', numel(imageYIdx));
seq.setDefinition('Nz_meas', numel(imageZIdx));
seq.setDefinition('ReadoutOversamplingFactor', os_factor);
seq.setDefinition('ReadoutDuration', Tread);
seq.setDefinition('ReadoutPolarity', 'monopolar');
seq.setDefinition('kSpaceCenterLine', floor(Ny/2));
seq.setDefinition('kSpaceCenterPartition', floor(Nz/2));
seq.setDefinition('PhaseResolution', (fov(1)/Nx)/(fov(2)/Ny));
seq.setDefinition('PartitionResolution', (fov(1)/Nx)/(fov(3)/Nz));
seq.setDefinition('ReceiverGainHigh', 1);
seq.setDefinition('WaveSinChannel', waveSinChannel);
seq.setDefinition('WaveCosChannel', waveCosChannel);
seq.setDefinition('WaveAmplitude_mTm', gwave_max);
seq.setDefinition('WaveSlew_Tms', swave_max);
seq.setDefinition('WaveCycles', Ncycles);
seq.setDefinition('UseFlowComp', double(isUseFlowComp));
seq.setDefinition('UseFullInitialFC', double(isFCInitialSine && isFCInitialReadout && ...
    isFCInitialCosine && isFCInitialPEY && isFCInitialPARZ && isFCSlabRephZ));
seq.setDefinition('UseFullInterEchoFC', double(Nechoes > 1 && ...
    isFCInterEchoReadoutCos && isFCInterEchoSine && ...
    isFCInterEchoPEY && isFCInterEchoPARZ));
seq.setDefinition('CalibrationTE', calibTE);
seq.setDefinition('CalibrationTR', calibTR);
seq.setDefinition('CalibrationNcalib1', Ncalib1);
seq.setDefinition('CalibrationNcalib2', Ncalib2);
seq.setDefinition('CalibrationNacs', NacsCal);
seq.setDefinition('CalibrationNSets', numel(calParts));
seq.setDefinition('CalibrationRefscanNLin', Ncalib1);
seq.setDefinition('CalibrationRefscanNPar', Ncalib1);
seq.setDefinition('CalibrationAllSetsInRefscan', 1);
seq.setDefinition('CalibrationACSSetID', 4);
seq.setDefinition('CalibrationSlabRephaserSeparate', 1);
seq.setDefinition('KspaceOrdering', 'negative_to_positive');

%% Compact sequence filename
% Keep the complete filename below the scanner interpreter's
% 128-character limit, including the version suffix and ".seq".
%
% Example:
% gre_3d_wave_FC_FOV220x220x180_res0p88x0p88x2p5_...
% E1_Ry3_Rz1_os4_amp8_cyc10_TRA_prisma_v151.seq

compactNum = @(x) strrep( ...
    strrep(sprintf('%.4g', x), '.', 'p'), ...
    '-', 'm');

fov_mm = fov(:).' * 1e3;
res_mm = fov(:).' ./ N(:).' * 1e3;

fovTokens = arrayfun(compactNum, fov_mm, ...
    'UniformOutput', false);
resTokens = arrayfun(compactNum, res_mm, ...
    'UniformOutput', false);

fovString = strjoin(fovTokens, 'x');
resString = strjoin(resTokens, 'x');

if isUseFlowComp
    fcString = 'FC';
else
    fcString = 'noFC';
end

if isUseWave_sin || isUseWave_cos
    sequenceName = 'gre_3d_wave';
else
    sequenceName = 'gre_3d_nowave';
end

seqBaseName = sprintf( ...
    ['%s_%s_FOV%s_res%s_E%d_' ...
     'Ry%d_Rz%d_os%d_amp%s_cyc%d_%s_%s'], ...
    sequenceName, ...
    fcString, ...
    fovString, ...
    resString, ...
    Nechoes, ...
    Ry, ...
    Rz, ...
    os_factor, ...
    compactNum(gwave_max), ...
    Ncycles, ...
    slOrientation, ...
    sys_type);

% Store the format-independent name in the sequence header.
seq.setDefinition('Name', seqBaseName);

%% Write sequence
outDir_v141 = fullfile(out_path, 'generated_seq_v141');
outDir_v151 = fullfile(out_path, 'generated_seq_v151');

if write_v141_format && ~exist(outDir_v141, 'dir')
    mkdir(outDir_v141);
end

if ~exist(outDir_v151, 'dir')
    mkdir(outDir_v151);
end

fileName_v141 = [seqBaseName '_v141.seq'];
fileName_v151 = [seqBaseName '_v151.seq'];

% The scanner filename must contain fewer than 128 characters.
assert(numel(fileName_v141) < 128, ...
    ['The v141 sequence filename contains %d characters. ' ...
     'It must contain fewer than 128 characters:\n%s'], ...
    numel(fileName_v141), fileName_v141);

assert(numel(fileName_v151) < 128, ...
    ['The v151 sequence filename contains %d characters. ' ...
     'It must contain fewer than 128 characters:\n%s'], ...
    numel(fileName_v151), fileName_v151);

if write_v141_format
    seqFile_v141 = fullfile(outDir_v141, fileName_v141);
    seq.write_v141(seqFile_v141);

    fprintf('Write to file (v141, %d filename characters):\n%s\n', ...
        numel(fileName_v141), seqFile_v141);
end

seqFile_v151 = fullfile(outDir_v151, fileName_v151);
seq.write(seqFile_v151);

fprintf('Write to file (v151, %d filename characters):\n%s\n', ...
    numel(fileName_v151), seqFile_v151);

%% Optional PNS/CNS and forbidden-frequency checks
% The sequence is deliberately written before these optional checks.
do_pns_check = promptYesNoFromWorkspace('do_pns_check', ...
    'Perform PNS/CNS check?', false);
if do_pns_check
    if isempty(safe_pns_prediction_path) || ~exist(safe_pns_prediction_path,'dir')
        fprintf('Skipping PNS/CNS check: safe_pns_prediction_path is unavailable.\n');
    elseif isempty(system_asc_file) || ~exist(system_asc_file,'file')
        fprintf('Skipping PNS/CNS check: system_asc_file is unavailable.\n');
    else
        warning('off','mr:restoreShape');
        try
            isHasCNS = strcmp(sys_type,'CimaX') || strcmp(sys_type,'TerraX');
            [~,tpns] = seq.calcPNS(system_asc_file,true,0);
            if ~isGEscanner && max(tpns)>0.95
                warning('PNS=%.2f too high; the sequence may not run.',max(tpns));
            end
            if isHasCNS
                [~,tpns] = seq.calcPNS(system_asc_file,true,1);
                if ~isGEscanner && max(tpns)>0.95
                    warning('CNS=%.2f too high; the sequence may not run.',max(tpns));
                end
            end
        catch ME
            warning('PNS/CNS check failed: %s',ME.message);
        end
        warning('on','mr:restoreShape');
    end
end

do_forbidden_frequency_check = promptYesNoFromWorkspace( ...
    'do_forbidden_frequency_check', 'Perform forbidden-frequency check?', false);
if do_forbidden_frequency_check
    if isGEscanner
        fprintf('Skipping forbidden-frequency check for premier/GE.\n');
    elseif isempty(system_asc_file) || ~exist(system_asc_file,'file')
        fprintf('Skipping forbidden-frequency check: system_asc_file is unavailable.\n');
    elseif exist('forbiddenFreqCheck','file') ~= 2
        fprintf('Skipping forbidden-frequency check: helper not found.\n');
    else
        warning('off','mr:restoreShape');
        try
            forbiddenFreqCheck(seq,sys,system_asc_file);
        catch ME
            warning('Forbidden-frequency check failed: %s',ME.message);
        end
        warning('on','mr:restoreShape');
    end
end

% Stop after normal generation/checks. Diagnostic sections below are kept
% for explicit Run Section use and do not execute automatically.
return;

%% Manual diagnostic: M0/M1 at image-center PE/PAR echo times
% This block is intentionally placed after the sequence write and after the
% return above. To run it, either select this section and Run Section in
% MATLAB, or temporarily comment out the return above. A second return at the
% end of this section prevents the older optional plotting blocks below from
% running unless you explicitly comment that second return too.
%
% It checks only image center PE/PAR readouts by default. If no non-REF image
% center readout exists, it falls back to center readouts including REF.
%
% M0/M1 are integrated from the selected reference point to each echo center,
% using full sequence gradients from seq.waveforms_and_times.
%
% Reference options:
%   'rfCenter'       : RF excitation center of the matching TR
%   'afterRF'        : end of the RF/slab-select block
%   'afterGzSsReph'  : after the ordinary center-line PE/slab rephaser block
%                      following RF, i.e. after gz_ss_reph/gzPreComb at
%                      center PAR. This excludes both slab-select and slab
%                      rephaser contributions from the moment window.
momRefMode = 'afterGzSsReph';   % choices: 'rfCenter', 'afterRF', 'afterGzSsReph'

warning('off', 'mr:restoreShape');
waveData_mom = seq.waveforms_and_times();
[ktraj_adc_mom, t_adc_mom, ktraj_mom, t_ktraj_mom, t_excitation_mom, t_refocusing_mom] = seq.calculateKspacePP(); %#ok<ASGLU>
warning('on', 'mr:restoreShape');

% Pull full-sequence gradient waveforms. Pulseq MATLAB returns gradients in
% Hz/m as piecewise-linear waveforms: row 1 = time [s], row 2 = gradient.
tGxFull = []; aGxFull = [];
tGyFull = []; aGyFull = [];
tGzFull = []; aGzFull = [];
if numel(waveData_mom) >= 1 && ~isempty(waveData_mom{1})
    tGxFull = waveData_mom{1}(1,:);
    aGxFull = waveData_mom{1}(2,:);
end
if numel(waveData_mom) >= 2 && ~isempty(waveData_mom{2})
    tGyFull = waveData_mom{2}(1,:);
    aGyFull = waveData_mom{2}(2,:);
end
if numel(waveData_mom) >= 3 && ~isempty(waveData_mom{3})
    tGzFull = waveData_mom{3}(1,:);
    aGzFull = waveData_mom{3}(2,:);
end

% ADC readout center times.
nAdcSamplesMom = numel(t_adc_mom);
adcSamplesPerReadoutMom = adc.numSamples;
assert(mod(nAdcSamplesMom, adcSamplesPerReadoutMom) == 0, ...
    'ADC sample count (%d) is not divisible by ADC samples/readout (%d).', ...
    nAdcSamplesMom, adcSamplesPerReadoutMom);
nReadoutsMom = nAdcSamplesMom / adcSamplesPerReadoutMom;
assert(mod(nReadoutsMom, Nechoes) == 0, ...
    'Readout count (%d) is not divisible by Nechoes (%d).', nReadoutsMom, Nechoes);
nExcWithAdcMom = nReadoutsMom / Nechoes;
assert(numel(t_excitation_mom) >= nExcWithAdcMom, ...
    'Not enough excitation timestamps to match ADC readouts.');
tExcUseMom = t_excitation_mom(end - nExcWithAdcMom + 1:end);

tAdc2dMom = reshape(t_adc_mom, adcSamplesPerReadoutMom, nReadoutsMom);
tEchoCenterMom = 0.5 * (tAdc2dMom(1,:) + tAdc2dMom(end,:));

% Match ADC labels to readouts.
adcLblMom = seq.evalLabels('evolution','adc');
linMom = adcLblMom.LIN(:).';
parMom = adcLblMom.PAR(:).';
assert(numel(linMom) == nReadoutsMom, ...
    'Label/readout mismatch: %d LIN labels but %d ADC readouts.', numel(linMom), nReadoutsMom);

if isfield(adcLblMom, 'ECO')
    ecoMom = adcLblMom.ECO(:).';
else
    ecoMom = mod(0:(nReadoutsMom-1), Nechoes);
end
if isfield(adcLblMom, 'REF')
    refMom = adcLblMom.REF(:).';
else
    refMom = zeros(1, nReadoutsMom);
end

centerLinLabel = floor(Ny/2);
centerParLabel = floor(Nz/2);
isCenterReadout = (linMom == centerLinLabel) & (parMom == centerParLabel);
isImageReadout = (refMom == 0);
selectedReadoutsMom = find(isCenterReadout & isImageReadout);
if isempty(selectedReadoutsMom)
    fprintf(['\nNo non-REF image center PE/PAR readouts found at LIN=%d, PAR=%d. ', ...
             'Falling back to all center readouts including REF, if present.\n'], ...
        centerLinLabel, centerParLabel);
    selectedReadoutsMom = find(isCenterReadout);
end

fprintf('\nFull-sequence M0/M1 diagnostic at image-center PE/PAR echo times:\n');
fprintf('  Source: seq.waveforms_and_times + seq.calculateKspacePP.\n');
fprintf('  Center labels: LIN=%d, PAR=%d. REF=0/non-ref preferred when REF exists.\n', ...
    centerLinLabel, centerParLabel);
fprintf('  Moment window: selected reference point -> ADC/readout center.\n');
fprintf('  M1 reference mode: %s. Units: M0 [1/m], M1 [1/m*s].\n', momRefMode);
fprintf('  Ordinary PE/PAR gradients are not excluded after the selected reference.\n');

if isempty(selectedReadoutsMom)
    fprintf('  No center PE/PAR readouts found. Check acceleration, labels, or center index definitions.\n\n');
else
    fprintf('  Checking %d center PE/PAR readouts.\n', numel(selectedReadoutsMom));
    fprintf('\n');
    fprintf('  Readout | Echo | LIN/PAR/REF | tRef(ms) | tEcho(ms) | M0x/M1x | M0y/M1y | M0z/M1z\n');
    for rr = selectedReadoutsMom(:).'
        excIdxMom = ceil(rr / Nechoes);
        if excIdxMom < 1 || excIdxMom > numel(tExcUseMom)
            error('Readout %d maps to invalid excitation index %d.', rr, excIdxMom);
        end
        tRfCenterMom = tExcUseMom(excIdxMom);
        switch lower(momRefMode)
            case {'rfcenter', 'rf_center', 'rf'}
                tRefMom = tRfCenterMom;
            case {'afterrf', 'after_rf', 'rfend', 'rf_end'}
                % End of the RF/slab-select block.
                tRefMom = tRfCenterMom + rfCenterToEndExc;
            case {'aftergzssreph', 'after_gz_ss_reph', 'aftergzreph', ...
                  'after_gz_reph', 'afterreph', 'after_reph'}
                % After the ordinary center-line prephaser block following RF.
                % When Step 9 merged PE-y FC is active, y/LIN PE is moved into
                % the right-edge-aligned prep/FC module, so the post-RF block
                % contains only gzPreComb for this diagnostic reference.
                if isUseMergedInitialYPreFC
                    centerPreBlockDurMom = mr.calcDuration(gzPreComb{zCenterIdx});
                else
                    centerPreBlockDurMom = max([ ...
                        mr.calcDuration(gyPre{yCenterIdx}), ...
                        mr.calcDuration(gzPreComb{zCenterIdx}) ...
                    ]);
                end
                tRefMom = tRfCenterMom + rfCenterToEndExc + centerPreBlockDurMom;
            otherwise
                error(['Unknown momRefMode = %s. Use ''rfCenter'', ''afterRF'', ', ...
                       'or ''afterGzSsReph''.'], momRefMode);
        end
        tEvalMom = tEchoCenterMom(rr);

        [M0xMom, M1xMom] = continuousMomentFromPolylineWindow(tGxFull, aGxFull, tRefMom, tEvalMom, tRefMom);
        [M0yMom, M1yMom] = continuousMomentFromPolylineWindow(tGyFull, aGyFull, tRefMom, tEvalMom, tRefMom);
        [M0zMom, M1zMom] = continuousMomentFromPolylineWindow(tGzFull, aGzFull, tRefMom, tEvalMom, tRefMom);

        fprintf('  %7d | %4d | %3d/%3d/%3d | %8.6f | %9.6f | %+.3e/%+.3e | %+.3e/%+.3e | %+.3e/%+.3e\n', ...
            rr, ecoMom(rr)+1, linMom(rr), parMom(rr), refMom(rr), tRefMom*1e3, tEvalMom*1e3, ...
            M0xMom, M1xMom, M0yMom, M1yMom, M0zMom, M1zMom);
    end
    fprintf('\n');
end

% Stop here when the manual M0/M1 section is enabled. Comment this return as
% well if you also want the older optional plotting/TE diagnostics below.
return;

%% Optional plotting and measured TE diagnostics

% seq.plot('TimeRange', [0 5*TR], 'label', 'lin,par,eco');
% seq.plot('TimeRange', [0 2*TR], 'stacked', 1);

warning('off', 'mr:restoreShape');
[ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing] = seq.calculateKspacePP();

nAdcSamples = numel(t_adc);
adcSamplesPerReadout = adc.numSamples;
assert(mod(nAdcSamples, adcSamplesPerReadout) == 0, ...
    'ADC sample count (%d) is not divisible by ADC samples/readout (%d).', nAdcSamples, adcSamplesPerReadout);
nReadouts = nAdcSamples / adcSamplesPerReadout;
assert(mod(nReadouts, Nechoes) == 0, 'Readout count (%d) is not divisible by Nechoes (%d).', nReadouts, Nechoes);
nExcWithAdc = nReadouts / Nechoes;
assert(numel(t_excitation) >= nExcWithAdc, 'Not enough excitation timestamps to match ADC readouts.');
t_exc_use = t_excitation(end - nExcWithAdc + 1:end);

t_adc_2d = reshape(t_adc, adcSamplesPerReadout, nReadouts);
t_ro_center = 0.5 * (t_adc_2d(1,:) + t_adc_2d(end,:));
t_ro_center_2d = reshape(t_ro_center, Nechoes, nExcWithAdc).';
TE_meas = t_ro_center_2d - t_exc_use(:);
TE_meas_mean = mean(TE_meas, 1);
TE_meas_std = std(TE_meas, 0, 1);

fprintf('\nMeasured TE from ADC-center relative to RF center (seq.calculateKspacePP):\n');
for c = 1:Nechoes
    fprintf('  Echo %d: mean = %.6f ms, std = %.6g ms, target = %.6f ms\n', ...
        c, TE_meas_mean(c)*1e3, TE_meas_std(c)*1e3, TE(c)*1e3);
end
TR_meas = diff(t_exc_use);
fprintf('Measured TR from excitation centers: mean = %.6f ms, std = %.6g ms, target = %.6f ms\n\n', ...
    mean(TR_meas)*1e3, std(TR_meas)*1e3, TR*1e3);
warning('on', 'mr:restoreShape');


return;

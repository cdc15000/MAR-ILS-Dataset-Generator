function status = run_mar_harness(varargin)
% RUN_MAR_HARNESS  MAR ILS laboratory harness, MATLAB (ASTM WK101378 draft Rev 06, v9 dataset).
%
%   Reads the coded sinograms of one replicate dataset in the format of the
%   sinogram ingestion specification (version 2.0), calls apply_mar() for
%   each, and writes one DICOM CT image per realization, named by its code,
%   with the CP-2575 MAR Macro (protocol §A8.3). Base MATLAB only (R2016b+);
%   no toolbox is needed, and the DICOM file is written byte by byte.
%
%   Reconstruction check, draft standard §12.2 (protocol Step 2.1):
%       run_mar_harness('--recon-check', 'FAMILIARIZATION/recon_check/recon_check.h5')
%
%   One run directory per assigned replicate dataset:
%       run_mar_harness('REP04', 'C2_R1')
%
%   Then produce the run directory's manifest (protocol §A8.2), for example
%   in a shell:  (cd C2_R1 && shasum -a 256 *.dcm) > C2_R1_checksums_sha256.txt
%
%   For Condition 2 (the laboratory's own MAR). Under Condition 1 the
%   harness is not used (protocol §A7).

if nargin == 2 && strcmp(varargin{1}, '--recon-check')
    status = recon_check(varargin{2});
elseif nargin == 2
    status = run(varargin{1}, varargin{2});
else
    error('Usage: run_mar_harness(dataset_dir, run_dir) or run_mar_harness(''--recon-check'', file)');
end
end


% ── Modes ────────────────────────────────────────────────────────────────

function status = recon_check(path)
[p, geo] = read_sinogram(path);
ref = h5read(path, '/reference_image_hu').';    % h5read reverses the axes
hu = reconstruct(p, geo);
worst = max(abs(hu(:) - ref(:)));
pass = worst <= 1.0;
verdict = 'FAIL';
if pass, verdict = 'PASS'; end
fprintf('Reconstruction check: largest difference %.6f HU (%s, tolerance 1.0 HU)\n', worst, verdict);
status = double(~pass);
end


function status = run(dataset_dir, run_dir)
files = dir(fullfile(dataset_dir, 'sinograms', '*.h5'));
if isempty(files)
    error('No sinograms in %s', fullfile(dataset_dir, 'sinograms'));
end
[~, order] = sort({files.name});
files = files(order);
if ~isfolder(run_dir), mkdir(run_dir); end
study_uid  = make_uid();
series_uid = make_uid();
fprintf('%s -> %s: %d realizations\n', dataset_dir, run_dir, numel(files));
for i = 1:numel(files)
    [~, code] = fileparts(files(i).name);
    [p, geo] = read_sinogram(fullfile(files(i).folder, files(i).name));
    hu = apply_mar(p, geo);
    if ~isequal(size(hu), [512 512])
        error('apply_mar returned size [%s] for %s, expected [512 512]', num2str(size(hu)), code);
    end
    write_ct_dicom(hu, fullfile(run_dir, [code '.dcm']), study_uid, series_uid);
    if mod(i, 100) == 0 || i == numel(files)
        fprintf('  %d/%d\n', i, numel(files));
    end
end
fprintf('Now write the manifest: (cd %s && shasum -a 256 *.dcm) > %s_checksums_sha256.txt\n', run_dir, run_dir);
status = 0;
end


% ── Ingestion spec §4 ────────────────────────────────────────────────────

function [p, geo] = read_sinogram(path)
% p is n_views x n_channels, p(k+1, j+1). h5read returns the HDF5 axes reversed.
ds = '/line_integrals';
p = double(h5read(path, ds)).';
geo.sid_mm          = double(h5readatt(path, ds, 'SID_mm'));
geo.sdd_mm          = double(h5readatt(path, ds, 'SDD_mm'));
geo.n_views         = double(h5readatt(path, ds, 'n_views'));
geo.n_channels      = double(h5readatt(path, ds, 'n_channels'));
geo.delta_gamma_rad = double(h5readatt(path, ds, 'delta_gamma_rad'));
geo.gamma0_rad      = double(h5readatt(path, ds, 'gamma0_rad'));
geo.betas_rad       = double(h5readatt(path, ds, 'betas_rad'));
geo.gammas_rad      = double(h5readatt(path, ds, 'gammas_rad'));
geo.I0              = double(h5readatt(path, ds, 'I0'));
geo.scatter_counts  = double(h5readatt(path, ds, 'scatter_counts'));
geo.electronic_variance_counts2 = double(h5readatt(path, ds, 'electronic_variance_counts2'));
geo.log_floor_counts = double(h5readatt(path, ds, 'log_floor_counts'));
if ~isequal(size(p), [geo.n_views geo.n_channels])
    error('%s: line_integrals is %s, expected %d x %d', path, mat2str(size(p)), geo.n_views, geo.n_channels);
end
end


% ── Minimal DICOM Part 10 writer (Explicit VR Little Endian) ─────────────

function write_ct_dicom(hu, path, study_uid, series_uid)
CT = '1.2.840.10008.5.1.4.1.1.2';
sop_uid = make_uid();

px = int16(max(min(round(hu), 32767), -32768));   % round to whole HU, signed 16-bit
px = px.';                                         % DICOM sends row 0 first, row-major

meta = [el(2, 1, 'OB', uint8([0 1])), ...
        el_str(2, 2, 'UI', CT), ...
        el_str(2, 3, 'UI', sop_uid), ...
        el_str(2, hex2dec('10'), 'UI', '1.2.840.10008.1.2.1'), ...
        el_str(2, hex2dec('12'), 'UI', '2.25.329800735698586629295641978511506172918')];
group_len = el(2, 0, 'UL', u32(numel(meta)));

item = el_str(hex2dec('18'), hex2dec('9391'), 'CS', 'YES');
item = [u16(hex2dec('FFFE')), u16(hex2dec('E000')), u32(numel(item)), item];
mar_seq = [u16(hex2dec('18')), u16(hex2dec('9390')), uint8('SQ'), u16(0), u32(numel(item)), item];

data = [el_str(8, hex2dec('16'), 'UI', CT), ...
        el_str(8, hex2dec('18'), 'UI', sop_uid), ...
        el_str(8, hex2dec('60'), 'CS', 'CT'), ...
        mar_seq, ...
        el_str(hex2dec('20'), hex2dec('0D'), 'UI', study_uid), ...
        el_str(hex2dec('20'), hex2dec('0E'), 'UI', series_uid), ...
        el_str(hex2dec('20'), hex2dec('13'), 'IS', '1'), ...
        el_str(hex2dec('20'), hex2dec('32'), 'DS', '-150\-150\0'), ...
        el_str(hex2dec('20'), hex2dec('37'), 'DS', '1\0\0\0\1\0'), ...
        el(hex2dec('28'), 2, 'US', u16(1)), ...
        el_str(hex2dec('28'), 4, 'CS', 'MONOCHROME2'), ...
        el(hex2dec('28'), hex2dec('10'), 'US', u16(512)), ...
        el(hex2dec('28'), hex2dec('11'), 'US', u16(512)), ...
        el_str(hex2dec('28'), hex2dec('30'), 'DS', '0.5859375\0.5859375'), ...
        el(hex2dec('28'), hex2dec('100'), 'US', u16(16)), ...
        el(hex2dec('28'), hex2dec('101'), 'US', u16(16)), ...
        el(hex2dec('28'), hex2dec('102'), 'US', u16(15)), ...
        el(hex2dec('28'), hex2dec('103'), 'US', u16(1)), ...
        el_str(hex2dec('28'), hex2dec('1052'), 'DS', '0'), ...
        el_str(hex2dec('28'), hex2dec('1053'), 'DS', '1'), ...
        el(hex2dec('7FE0'), hex2dec('10'), 'OW', le_bytes(px(:)))];

fid = fopen(path, 'w');
if fid < 0, error('Cannot write %s', path); end
fwrite(fid, [zeros(1, 128, 'uint8'), uint8('DICM'), group_len, meta, data], 'uint8');
fclose(fid);
end


function b = el(grp, elm, vr, data)
% One element; data is a uint8 row vector of even length.
long_form = any(strcmp(vr, {'OB', 'OW', 'SQ', 'UN', 'UT'}));
if long_form
    b = [u16(grp), u16(elm), uint8(vr), u16(0), u32(numel(data)), data];
else
    b = [u16(grp), u16(elm), uint8(vr), u16(numel(data)), data];
end
end


function b = el_str(grp, elm, vr, s)
% Text element padded to even length: UI with a null byte, other VRs with a space.
s = uint8(s);
if mod(numel(s), 2)
    if strcmp(vr, 'UI'), s = [s uint8(0)]; else, s = [s uint8(' ')]; end
end
b = el(grp, elm, vr, s);
end


function b = u16(v), b = le_bytes(uint16(v)); end
function b = u32(v), b = le_bytes(uint32(v)); end


function b = le_bytes(x)
% Little-endian bytes of an integer array, as a row.
b = typecast(x(:).', 'uint8');
[~, ~, endian] = computer;
if endian == 'B'
    n = numel(typecast(x(1), 'uint8'));
    b = reshape(flipud(reshape(b, n, [])), 1, []);
end
end


function uid = make_uid()
% 2.25 UID: a random integer below 2^128 (38 digits, first digit nonzero).
% Uses its own clock-seeded stream, so the global random state of the
% laboratory's code is left alone and UIDs differ between sessions.
persistent rs
if isempty(rs), rs = RandStream('mt19937ar', 'Seed', 'shuffle'); end
d = [randi(rs, [1 9]), randi(rs, [0 9], 1, 37)];
uid = ['2.25.' char('0' + d)];
end

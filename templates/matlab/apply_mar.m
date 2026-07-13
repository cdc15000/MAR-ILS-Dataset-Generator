function hu_volume = apply_mar(sinogram, geo)
% APPLY_MAR  Lab MAR Implementation — REPLACE THIS FILE
%
%   This file contains the function that the harness calls for each
%   realization.  Replace the body with your reconstruction + MAR pipeline.
%
%   The example below performs plain FBP (no MAR) as a starting point.
%
%   Args:
%     sinogram — single array, size [n_slices, n_angles, n_det]
%                Line integrals in neper (ready to reconstruct).
%     geo      — struct with fields:
%                  sid_mm, sdd_mm, n_slices, n_angles, n_det, voxel_mm,
%                  angles_deg (720x1), det_fan_angles_deg (512x1),
%                  gamma_max_deg, delta_gamma_deg
%
%   Returns:
%     hu_volume — single array, size [n_slices, 512, 512]
%                 Reconstructed image in Hounsfield Units.

% ── REPLACE EVERYTHING BELOW WITH YOUR MAR PIPELINE ──

n_slices = geo.n_slices;
n_angles = geo.n_angles;
n_det    = geo.n_det;
nx = 512; ny = 512;

sid_vox     = geo.sid_mm / geo.voxel_mm;
sid_cm      = geo.sid_mm / 10.0;
angles_rad  = deg2rad(geo.angles_deg(:)');
det_fan_rad = deg2rad(geo.det_fan_angles_deg(:)');
delta_gamma = deg2rad(geo.delta_gamma_deg);
cos_fan     = cos(det_fan_rad);
gamma_min   = det_fan_rad(1);
cx = nx / 2.0;
cy = ny / 2.0;

mu_tissue     = 0.2059;
dc_offset     = -0.029;
background_hu = 40.0;
scale_factor  = pi / n_angles / (sid_cm * delta_gamma);

hu_volume = zeros(n_slices, ny, nx, 'single');

for z = 1:n_slices
    sino = double(squeeze(sinogram(z, :, :)));  % [n_angles, n_det]

    % 1. Cosine pre-weight
    weighted = sino .* cos_fan;

    % 2. Ram-Lak filter
    freq = (0:floor(n_det/2)) / n_det;
    ramp = abs(freq);
    F = fft(weighted, n_det, 2);
    F(:, 1:length(ramp)) = F(:, 1:length(ramp)) .* ramp;
    F(:, n_det:-1:n_det-length(ramp)+2) = ...
        F(:, n_det:-1:n_det-length(ramp)+2) .* ramp(2:end);
    filtered = real(ifft(F, n_det, 2));

    % 3. Distance-weighted backprojection
    recon = zeros(ny, nx);
    [ix_grid, iy_grid] = meshgrid(0:(nx-1), 0:(ny-1));

    for a = 1:n_angles
        beta = angles_rad(a);
        sx = cx + sid_vox * cos(beta);
        sy = cy + sid_vox * sin(beta);

        dx = ix_grid - sx;
        dy = iy_grid - sy;
        L_sq = dx.^2 + dy.^2;

        pixel_angle = atan2(dy, dx);
        cr_angle = beta + pi;
        gamma = atan2(sin(pixel_angle - cr_angle), cos(pixel_angle - cr_angle));

        di = (gamma - gamma_min) / delta_gamma;
        di_floor = floor(di);
        frac = di - di_floor;
        di1 = di_floor + 1;  % MATLAB 1-indexed
        di2 = di1 + 1;

        valid = (di1 >= 1) & (di2 <= n_det);
        val = zeros(ny, nx);
        val(valid) = (1 - frac(valid)) .* filtered(a, di1(valid)) + ...
                     frac(valid) .* filtered(a, di2(valid));
        recon = recon + val .* (sid_vox^2) ./ L_sq;
    end

    % 4. Scale and convert to HU
    mu_recon = recon * scale_factor;
    hu = single((mu_recon - mu_tissue - dc_offset) / mu_tissue * 1000.0 ...
                + background_hu);
    hu_volume(z, :, :) = hu;
end

end

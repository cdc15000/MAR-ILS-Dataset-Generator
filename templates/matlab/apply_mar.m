function hu = apply_mar(p, geo)
% APPLY_MAR  Laboratory MAR implementation: REPLACE THE BODY.
%
%   hu = apply_mar(p, geo)
%
%   p    double, 984 x 888, line integrals: p(k+1, j+1) for view k and
%        channel j (ingestion spec §4.1). Counts, if needed: geo.I0 * exp(-p).
%   geo  struct with sid_mm, sdd_mm, n_views, n_channels, delta_gamma_rad,
%        gamma0_rad, betas_rad, gammas_rad, I0, scatter_counts,
%        electronic_variance_counts2, log_floor_counts.
%   hu   512 x 512 image in HU (§A1.1(h)): hu(r+1, c+1) for row r and
%        column c, pixel (256, 256) at the isocenter (ingestion spec §6.2).
%
%   As supplied, no MAR is applied: the output matches the supplied noMAR
%   image of the same realization to within DICOM rounding. Run it once that
%   way to check the harness before plugging in your method.
%
%   These templates serve Condition 2 (the laboratory's own MAR). Under
%   Condition 1 the laboratory works from the text of the standard and the
%   ingestion specification only (protocol §A7).

% ── REPLACE WITH YOUR MAR PIPELINE ──
% A sinogram-domain method corrects p here and then calls reconstruct().
hu = reconstruct(p, geo);
end

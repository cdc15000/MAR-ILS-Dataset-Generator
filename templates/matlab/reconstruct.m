function hu = reconstruct(p, geo)
% RECONSTRUCT  Reconstruction used under draft standard §14.3, in HU (512 x 512).
%
%   As supplied, the reference reconstruction. If you use your own, put it
%   here; it must pass run_mar_harness('--recon-check', ...).

MU_WATER_CM = 0.2124;                  % §A1.1(h)
mu = reference_fbp(p, geo);
hu = 1000 * (mu - MU_WATER_CM) / MU_WATER_CM;
end

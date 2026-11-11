function mu = reference_fbp(p, geo)
% REFERENCE_FBP  Reference reconstruction, ASTM WK101378 draft Rev 06, §A1.1(g) and (h).
%
%   mu = reference_fbp(p, geo) reconstructs one sinogram p (n_views x n_channels,
%   indexed p(k+1, j+1) for view k and channel j) to a 512 x 512 image of mu in
%   cm^-1, indexed mu(r+1, c+1) for row r and column c (ingestion spec §6.2).
%
%   Equi-angular fan-beam filtered backprojection: projections weighted by
%   D cos(gamma), the Ram-Lak kernel sampled in the spatial domain at the
%   channel spacing, linear convolution, and 1/L^2 weighted pixel-driven
%   backprojection with linear interpolation between adjacent channels.
%   Coordinates follow §A1.1(i). Lengths are in cm, so mu comes out in cm^-1.
%
%   For Condition 2 of the ILS. Under Condition 1 laboratories do not consult
%   or reuse this file (protocol §A7). Requires R2016b or later.

nv  = double(geo.n_views);
nc  = double(geo.n_channels);
dg  = geo.delta_gamma_rad;
sid = geo.sid_mm / 10;                 % cm
pix = 30 / 512;                        % cm

% Ram-Lak kernel for equi-angular rays (Kak and Slaney §3.4.2), n = -(nc-1)..(nc-1)
n = -(nc - 1):(nc - 1);
g = zeros(size(n));
g(n == 0) = 1 / (8 * dg^2);
odd = mod(n, 2) ~= 0;
g(odd) = -1 ./ (2 * (pi * sin(n(odd) * dg)).^2);

% Weight by D cos(gamma) and convolve each view, times d_gamma
w = double(p) .* (sid * cos(geo.gammas_rad(:)'));
qfull = conv2(w, g);                   % nv x (3*nc - 2)
q = qfull(:, nc:(2 * nc - 1)) * dg;    % output for channels 0..nc-1

% Pixel-driven backprojection
idx = (0:511) - 256;
[X, Y] = meshgrid(idx * pix, -idx * pix);   % X(r+1,c+1) = x of column c, Y = y of row r
acc = zeros(512);
for k = 1:nv
    cb = cos(geo.betas_rad(k));
    sb = sin(geo.betas_rad(k));
    u = sid - (X * cb + Y * sb);
    v = X * sb - Y * cb;
    di = (atan2(v, u) - geo.gamma0_rad) / dg;
    ok = di >= 0 & di <= nc - 1;
    d  = di(ok);
    lo = min(floor(d), nc - 2);
    f  = d - lo;
    qk = q(k, :).';                    % column, so qk(lo + 1) is a column
    acc(ok) = acc(ok) + ((1 - f) .* qk(lo + 1) + f .* qk(lo + 2)) ./ (u(ok).^2 + v(ok).^2);
end
mu = acc * (2 * pi / nv);
end

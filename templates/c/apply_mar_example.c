/*
 * Lab MAR Implementation — REPLACE THIS FILE
 * ASTM WKXXXXX Rev 05
 *
 * Example: plain FBP (no MAR) as a starting point.
 * Replace the body of apply_mar() with your reconstruction + MAR pipeline.
 *
 * For CUDA: implement apply_mar() as a host function that launches your
 * kernels.  The harness links against this file — the CUDA compilation
 * is your responsibility.
 */

#include "apply_mar.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NX 512
#define NY 512

int apply_mar(const float *sinogram,
              const FanBeamGeometry *geo,
              float *hu_out)
{
    /* ── REPLACE EVERYTHING BELOW WITH YOUR MAR PIPELINE ── */

    const int n_slices = geo->n_slices;
    const int n_angles = geo->n_angles;
    const int n_det    = geo->n_det;

    const double sid_vox     = geo->sid_mm / geo->voxel_mm;
    const double sid_cm      = geo->sid_mm / 10.0;
    const double delta_gamma = geo->delta_gamma_deg * M_PI / 180.0;
    const double gamma_min   = geo->det_fan_angles_deg[0] * M_PI / 180.0;
    const double cx = NX / 2.0;
    const double cy = NY / 2.0;

    const double mu_tissue     = 0.2059;
    const double dc_offset     = -0.029;
    const double background_hu = 40.0;
    const double scale_factor  = M_PI / n_angles / (sid_cm * delta_gamma);

    /* Pre-compute angle arrays in radians */
    double *angles_rad = (double *)malloc(n_angles * sizeof(double));
    double *det_fan_rad = (double *)malloc(n_det * sizeof(double));
    double *cos_fan = (double *)malloc(n_det * sizeof(double));
    if (!angles_rad || !det_fan_rad || !cos_fan) return -1;

    for (int a = 0; a < n_angles; a++)
        angles_rad[a] = geo->angles_deg[a] * M_PI / 180.0;
    for (int d = 0; d < n_det; d++) {
        det_fan_rad[d] = geo->det_fan_angles_deg[d] * M_PI / 180.0;
        cos_fan[d] = cos(det_fan_rad[d]);
    }

    /* Per-slice work buffers */
    double *weighted = (double *)malloc(n_angles * n_det * sizeof(double));
    double *filtered = (double *)malloc(n_angles * n_det * sizeof(double));
    double *recon    = (double *)malloc(NY * NX * sizeof(double));
    if (!weighted || !filtered || !recon) return -1;

    for (int z = 0; z < n_slices; z++) {
        const float *sino_z = sinogram + (size_t)z * n_angles * n_det;

        /* 1. Cosine pre-weight */
        for (int a = 0; a < n_angles; a++)
            for (int d = 0; d < n_det; d++)
                weighted[a * n_det + d] = (double)sino_z[a * n_det + d] * cos_fan[d];

        /*
         * 2. Ram-Lak filter (simplified spatial-domain convolution).
         *    For production, use FFT-based filtering.
         */
        memcpy(filtered, weighted, n_angles * n_det * sizeof(double));

        /* 3. Distance-weighted backprojection */
        memset(recon, 0, NY * NX * sizeof(double));

        for (int a = 0; a < n_angles; a++) {
            double beta = angles_rad[a];
            double sb = sin(beta), cb = cos(beta);
            double sx = cx + sid_vox * cb;
            double sy = cy + sid_vox * sb;

            for (int iy = 0; iy < NY; iy++) {
                for (int ix = 0; ix < NX; ix++) {
                    double dx_p = (double)ix - sx;
                    double dy_p = (double)iy - sy;
                    double L_sq = dx_p * dx_p + dy_p * dy_p;

                    double pixel_angle = atan2(dy_p, dx_p);
                    double cr_angle = beta + M_PI;
                    double diff = pixel_angle - cr_angle;
                    double gamma = atan2(sin(diff), cos(diff));

                    double di = (gamma - gamma_min) / delta_gamma;
                    int di_int = (int)di;
                    if (di_int >= 0 && di_int < n_det - 1) {
                        double frac = di - di_int;
                        double val = (1.0 - frac) * filtered[a * n_det + di_int]
                                   + frac * filtered[a * n_det + di_int + 1];
                        recon[iy * NX + ix] += val * (sid_vox * sid_vox) / L_sq;
                    }
                }
            }
        }

        /* 4. Scale and convert to HU */
        float *hu_z = hu_out + (size_t)z * NY * NX;
        for (int i = 0; i < NY * NX; i++) {
            double mu_r = recon[i] * scale_factor;
            hu_z[i] = (float)(
                (mu_r - mu_tissue - dc_offset) / mu_tissue * 1000.0
                + background_hu
            );
        }
    }

    free(angles_rad);
    free(det_fan_rad);
    free(cos_fan);
    free(weighted);
    free(filtered);
    free(recon);

    return 0;
}

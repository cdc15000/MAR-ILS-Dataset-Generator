/*
 * Reference reconstruction of ASTM WK101378 draft Rev 06, §A1.1(g) and (h).
 *
 * Equi-angular fan-beam filtered backprojection: projections weighted by
 * D cos(gamma), the Ram-Lak kernel sampled in the spatial domain at the
 * channel spacing, linear convolution (direct sum, so no padding is needed),
 * and 1/L^2 weighted pixel-driven backprojection with linear interpolation
 * between adjacent channels. Coordinates follow §A1.1(i). Lengths are in cm,
 * so mu comes out in cm^-1.
 *
 * For Condition 2 of the ILS. Under Condition 1 laboratories do not consult
 * or reuse this file (protocol §A7).
 *
 * Build with -fopenmp to parallelize the backprojection over rows.
 */
#include "reference_fbp.h"

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FOV_CM 30.0
#define CENTER_PX 256

int reference_fbp(const double *p, const Geometry *geo, double *mu_out)
{
    const int nv = geo->n_views, nc = geo->n_channels;
    const double dg = geo->delta_gamma_rad;
    const double sid = geo->sid_mm / 10.0;          /* cm */
    const double pix = FOV_CM / IMAGE_PX;           /* cm */

    double *g = malloc((size_t)(2 * nc - 1) * sizeof(double));   /* g[n + nc - 1], n = -(nc-1)..nc-1 */
    double *w = malloc((size_t)nc * sizeof(double));
    double *q = malloc((size_t)nv * nc * sizeof(double));
    if (!g || !w || !q) { free(g); free(w); free(q); return -1; }

    /* Ram-Lak kernel for equi-angular rays (Kak and Slaney §3.4.2) */
    for (int n = -(nc - 1); n <= nc - 1; n++) {
        double v = 0.0;
        if (n == 0) v = 1.0 / (8.0 * dg * dg);
        else if (n % 2 != 0) { double s = M_PI * sin(n * dg); v = -1.0 / (2.0 * s * s); }
        g[n + nc - 1] = v;
    }

    /* Weight by D cos(gamma) and convolve, times d_gamma */
    for (int k = 0; k < nv; k++) {
        const double *pk = p + (size_t)k * nc;
        for (int j = 0; j < nc; j++) w[j] = pk[j] * sid * cos(geo->gammas_rad[j]);
        for (int j = 0; j < nc; j++) {
            double acc = 0.0;
            for (int m = 0; m < nc; m++) acc += w[m] * g[j - m + nc - 1];
            q[(size_t)k * nc + j] = acc * dg;
        }
    }

    /* Backprojection */
    const double dbeta = 2.0 * M_PI / nv;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < IMAGE_PX; r++) {
        const double y = (CENTER_PX - r) * pix;
        for (int c = 0; c < IMAGE_PX; c++) {
            const double x = (c - CENTER_PX) * pix;
            double acc = 0.0;
            for (int k = 0; k < nv; k++) {
                const double cb = cos(geo->betas_rad[k]), sb = sin(geo->betas_rad[k]);
                const double u = sid - (x * cb + y * sb);
                const double v = x * sb - y * cb;
                const double di = (atan2(v, u) - geo->gamma0_rad) / dg;
                if (di >= 0.0 && di <= nc - 1) {
                    int lo = (int)di;
                    if (lo > nc - 2) lo = nc - 2;
                    const double f = di - lo;
                    const double *qk = q + (size_t)k * nc;
                    acc += ((1.0 - f) * qk[lo] + f * qk[lo + 1]) / (u * u + v * v);
                }
            }
            mu_out[r * IMAGE_PX + c] = acc * dbeta;
        }
    }

    free(g); free(w); free(q);
    return 0;
}

void mu_to_hu(double *img, int n)
{
    for (int i = 0; i < n; i++) img[i] = 1000.0 * (img[i] - MU_WATER_CM) / MU_WATER_CM;
}

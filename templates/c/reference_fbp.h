/*
 * Reference reconstruction of ASTM WK101378 draft Rev 06, §A1.1(g) and (h).
 */
#ifndef REFERENCE_FBP_H
#define REFERENCE_FBP_H

#include "apply_mar.h"

#define MU_WATER_CM 0.2124   /* §A1.1(h) */

/* Reconstruct p (n_views x n_channels) to mu in cm^-1, mu_out[r * 512 + c]. Returns 0 on success. */
int reference_fbp(const double *p, const Geometry *geo, double *mu_out);

/* HU per §A1.1(h), in place on n values. */
void mu_to_hu(double *img, int n);

#endif /* REFERENCE_FBP_H */

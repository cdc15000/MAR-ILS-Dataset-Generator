/*
 * Laboratory MAR implementation: COPY THIS FILE AND REPLACE THE BODIES.
 *
 * As supplied, apply_mar() applies no MAR: it reconstructs the sinogram with
 * the reference reconstruction, so its output matches the supplied noMAR
 * image of the same realization to within DICOM rounding. Run it once that
 * way to check the harness before plugging in your method.
 *
 * For CUDA: implement these as host functions that launch your kernels,
 * compile the .cu files separately and link them with the harness.
 */
#include "apply_mar.h"
#include "reference_fbp.h"

int reconstruct(const double *p, const Geometry *geo, double *hu_out)
{
    if (reference_fbp(p, geo, hu_out) != 0) return -1;
    mu_to_hu(hu_out, IMAGE_PX * IMAGE_PX);
    return 0;
}

int apply_mar(const double *p, const Geometry *geo, double *hu_out)
{
    /* ── REPLACE WITH YOUR MAR PIPELINE ──
     * A sinogram-domain method corrects a copy of p here and then calls
     * reconstruct(). */
    return reconstruct(p, geo, hu_out);
}

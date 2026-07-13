/*
 * MAR ILS Lab Interface — C
 * ASTM WKXXXXX Rev 05
 *
 * The lab implements apply_mar() in their own source file.
 * The harness calls it once per realization.
 */

#ifndef APPLY_MAR_H
#define APPLY_MAR_H

#include <stdint.h>

typedef struct {
    int32_t  n_slices;          /* 256                                      */
    int32_t  n_angles;          /* 720                                      */
    int32_t  n_det;             /* 512                                      */
    double   voxel_mm;          /* 0.5                                      */
    double   sid_mm;            /* 570.0  source-to-isocenter distance      */
    double   sdd_mm;            /* 1040.0 source-to-detector distance       */
    double   gamma_max_deg;     /* ~12.97 half fan-angle                    */
    double   delta_gamma_deg;   /* ~0.0507 detector angular spacing         */
    double  *angles_deg;        /* [n_angles] source rotation angles        */
    double  *det_fan_angles_deg;/* [n_det] fan angle per detector element   */
} FanBeamGeometry;

/*
 * apply_mar — Lab's MAR reconstruction function.
 *
 * Parameters:
 *   sinogram  — Input line integrals in neper (float32).
 *               Layout: sinogram[z * n_angles * n_det + a * n_det + d]
 *               where z=slice, a=angle, d=detector.
 *               Shape: (n_slices, n_angles, n_det) row-major.
 *
 *   geo       — Fan-beam acquisition geometry (all parameters needed
 *               to reconstruct).
 *
 *   hu_out    — Output buffer, pre-allocated by the harness.
 *               Layout: hu_out[z * 512 * 512 + y * 512 + x]
 *               Shape: (n_slices, 512, 512) row-major.
 *               Values must be in Hounsfield Units (int16 range).
 *
 * Returns:
 *   0 on success, non-zero on error.
 */
int apply_mar(const float *sinogram,
              const FanBeamGeometry *geo,
              float *hu_out);

#endif /* APPLY_MAR_H */

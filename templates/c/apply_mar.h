/*
 * MAR ILS laboratory interface, C (ASTM WK101378 draft Rev 06, v9 dataset).
 *
 * The laboratory implements apply_mar() and reconstruct() in its own source
 * file (see apply_mar_example.c). The harness calls apply_mar() once per
 * realization and reconstruct() for the reconstruction check (§12.2).
 *
 * For Condition 2 (the laboratory's own MAR). Under Condition 1 the harness
 * and the reference reconstruction are not used (protocol §A7).
 */
#ifndef APPLY_MAR_H
#define APPLY_MAR_H

#include <stdint.h>

#define IMAGE_PX 512

/* Attributes of /line_integrals (ingestion spec §4.2). Arrays are owned by the harness. */
typedef struct {
    double   sid_mm;                       /* 541.0                         */
    double   sdd_mm;                       /* 949.0                         */
    int32_t  n_views;                      /* 984                           */
    int32_t  n_channels;                   /* 888                           */
    double   delta_gamma_rad;              /* 0.1/94.9                      */
    double   gamma0_rad;                   /* fan angle of channel 0        */
    double  *betas_rad;                    /* [n_views], 2*pi*k/984         */
    double  *gammas_rad;                   /* [n_channels], (j-443.5)*dg    */
    double   I0;                           /* 1e6                           */
    double   scatter_counts;               /* 100                           */
    double   electronic_variance_counts2;  /* 160                           */
    double   log_floor_counts;             /* 1                             */
} Geometry;

/*
 * apply_mar: apply MAR to one realization.
 *
 *   p       line integrals, p[k * n_channels + j] for view k and channel j
 *           (ingestion spec §4.1). Counts, if needed: I0 * exp(-p).
 *   geo     acquisition geometry
 *   hu_out  output, pre-allocated by the harness: hu_out[r * 512 + c] in HU
 *           (§A1.1(h)) for row r and column c, pixel (256, 256) at the
 *           isocenter (ingestion spec §6.2)
 *
 * Returns 0 on success.
 */
int apply_mar(const double *p, const Geometry *geo, double *hu_out);

/*
 * reconstruct: the reconstruction used under draft standard §14.3, in HU,
 * same layout as apply_mar. The supplied example uses the reference
 * reconstruction; your own must pass `mar_harness --recon-check`.
 */
int reconstruct(const double *p, const Geometry *geo, double *hu_out);

#endif /* APPLY_MAR_H */

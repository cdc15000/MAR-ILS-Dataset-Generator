/*
 * MAR ILS laboratory harness, C (ASTM WK101378 draft Rev 06, v9 dataset).
 *
 * Reads the coded sinograms of one replicate dataset in the format of the
 * sinogram ingestion specification (version 2.0), calls apply_mar() for
 * each, and writes one DICOM CT image per realization, named by its code,
 * with the CP-2575 MAR Macro (protocol §A8.3).
 *
 * Usage:
 *   ./mar_harness --recon-check FAMILIARIZATION/recon_check/recon_check.h5
 *   ./mar_harness REP04 C2_R1
 *
 * Then produce the run directory's manifest (protocol §A8.2), for example:
 *   (cd C2_R1 && shasum -a 256 *.dcm) > C2_R1_checksums_sha256.txt
 *
 * Dependencies: libhdf5. DICOM output uses a minimal Part 10 writer.
 */
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "hdf5.h"
#include "apply_mar.h"

#define RECON_CHECK_TOL_HU 1.0
#define NPIX (IMAGE_PX * IMAGE_PX)

/* ── Ingestion spec §4 ─────────────────────────────────────────────────── */

static int attr_double(hid_t obj, const char *name, double *out)
{
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0) { fprintf(stderr, "missing attribute %s\n", name); return -1; }
    herr_t rc = H5Aread(a, H5T_NATIVE_DOUBLE, out);
    H5Aclose(a);
    return rc < 0 ? -1 : 0;
}

static int attr_int(hid_t obj, const char *name, int32_t *out)
{
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    if (a < 0) { fprintf(stderr, "missing attribute %s\n", name); return -1; }
    herr_t rc = H5Aread(a, H5T_NATIVE_INT32, out);
    H5Aclose(a);
    return rc < 0 ? -1 : 0;
}

static void free_geometry(Geometry *g)
{
    free(g->betas_rad);
    free(g->gammas_rad);
    g->betas_rad = g->gammas_rad = NULL;
}

/* Reads /line_integrals as double, p[k * n_channels + j], and its attributes. */
static int read_sinogram(const char *path, double **p_out, Geometry *g)
{
    memset(g, 0, sizeof(*g));
    *p_out = NULL;
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (f < 0) return -1;
    hid_t d = H5Dopen2(f, "/line_integrals", H5P_DEFAULT);
    if (d < 0) { H5Fclose(f); return -1; }

    int rc = 0;
    rc |= attr_double(d, "SID_mm", &g->sid_mm);
    rc |= attr_double(d, "SDD_mm", &g->sdd_mm);
    rc |= attr_int(d, "n_views", &g->n_views);
    rc |= attr_int(d, "n_channels", &g->n_channels);
    rc |= attr_double(d, "delta_gamma_rad", &g->delta_gamma_rad);
    rc |= attr_double(d, "gamma0_rad", &g->gamma0_rad);
    rc |= attr_double(d, "I0", &g->I0);
    rc |= attr_double(d, "scatter_counts", &g->scatter_counts);
    rc |= attr_double(d, "electronic_variance_counts2", &g->electronic_variance_counts2);
    rc |= attr_double(d, "log_floor_counts", &g->log_floor_counts);

    hid_t space = H5Dget_space(d);
    hsize_t dims[2] = {0, 0};
    if (rc == 0 && (H5Sget_simple_extent_ndims(space) != 2 ||
                    H5Sget_simple_extent_dims(space, dims, NULL) < 0 ||
                    dims[0] != (hsize_t)g->n_views || dims[1] != (hsize_t)g->n_channels)) {
        fprintf(stderr, "%s: line_integrals is not (n_views, n_channels)\n", path);
        rc = -1;
    }
    H5Sclose(space);

    if (rc == 0) {
        g->betas_rad = malloc((size_t)g->n_views * sizeof(double));
        g->gammas_rad = malloc((size_t)g->n_channels * sizeof(double));
        *p_out = malloc((size_t)g->n_views * g->n_channels * sizeof(double));
        if (!g->betas_rad || !g->gammas_rad || !*p_out) rc = -1;
    }
    if (rc == 0) {
        rc |= attr_double(d, "betas_rad", g->betas_rad);
        rc |= attr_double(d, "gammas_rad", g->gammas_rad);
        /* stored as float32; HDF5 converts to double on read */
        if (H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, *p_out) < 0) rc = -1;
    }
    H5Dclose(d);
    H5Fclose(f);
    if (rc != 0) { free(*p_out); *p_out = NULL; free_geometry(g); }
    return rc;
}

/* ── Minimal DICOM Part 10 writer (Explicit VR Little Endian) ──────────── */

typedef struct { uint8_t *b; size_t n; } Buf;

static void put16(Buf *o, uint16_t v) { o->b[o->n++] = v & 0xFF; o->b[o->n++] = v >> 8; }
static void put32(Buf *o, uint32_t v) { put16(o, v & 0xFFFF); put16(o, v >> 16); }
static void set32(uint8_t *at, uint32_t v) { at[0] = v; at[1] = v >> 8; at[2] = v >> 16; at[3] = v >> 24; }

/* Elements must be written in ascending tag order. */
static void elem(Buf *o, uint16_t grp, uint16_t el, const char *vr, const void *data, uint32_t len)
{
    put16(o, grp); put16(o, el);
    o->b[o->n++] = vr[0]; o->b[o->n++] = vr[1];
    int long_form = !strcmp(vr, "OB") || !strcmp(vr, "OW") || !strcmp(vr, "SQ") ||
                    !strcmp(vr, "UN") || !strcmp(vr, "UT");
    if (long_form) { put16(o, 0); put32(o, len); } else put16(o, (uint16_t)len);
    if (len) { memcpy(o->b + o->n, data, len); o->n += len; }
}

/* Text element padded to even length: UI with a null byte, other VRs with a space. */
static void elem_str(Buf *o, uint16_t grp, uint16_t el, const char *vr, const char *s)
{
    char tmp[256];
    size_t len = strlen(s);
    memcpy(tmp, s, len);
    if (len % 2) tmp[len++] = strcmp(vr, "UI") ? ' ' : '\0';
    elem(o, grp, el, vr, tmp, (uint32_t)len);
}

static void elem_us(Buf *o, uint16_t grp, uint16_t el, uint16_t v)
{
    uint8_t b[2] = { v & 0xFF, v >> 8 };
    elem(o, grp, el, "US", b, 2);
}

static void make_uid(char *out, size_t size)
{
    uint64_t r[2] = {0, 0};
    FILE *u = fopen("/dev/urandom", "rb");
    if (!u || fread(r, sizeof(r), 1, u) != 1) { r[0] = (uint64_t)rand() + 1; r[1] = (uint64_t)rand(); }
    if (u) fclose(u);
    if (r[0] == 0) r[0] = 1;
    /* 2.25 UIDs are integers below 2^128: at most 20 + 18 digits here */
    snprintf(out, size, "2.25.%llu%018llu", (unsigned long long)r[0],
             (unsigned long long)(r[1] % 1000000000000000000ULL));
}

static int write_dicom(const double *hu, const char *path, const char *study_uid, const char *series_uid)
{
    static const char *CT = "1.2.840.10008.5.1.4.1.1.2";
    static const char *EVRLE = "1.2.840.10008.1.2.1";
    char sop_uid[96];
    make_uid(sop_uid, sizeof(sop_uid));

    Buf o = { calloc(1, (size_t)NPIX * 2 + 4096), 0 };
    int16_t *px = malloc((size_t)NPIX * sizeof(int16_t));
    if (!o.b || !px) { free(o.b); free(px); return -1; }
    for (int i = 0; i < NPIX; i++) {
        double v = rint(hu[i]);
        if (v < -32768.0) v = -32768.0;
        if (v > 32767.0) v = 32767.0;
        px[i] = (int16_t)v;
    }

    o.n = 128;
    memcpy(o.b + o.n, "DICM", 4); o.n += 4;

    /* File meta information */
    size_t glen_at = o.n + 8;
    elem(&o, 0x0002, 0x0000, "UL", "\0\0\0\0", 4);
    size_t meta_start = o.n;
    elem(&o, 0x0002, 0x0001, "OB", "\0\1", 2);
    elem_str(&o, 0x0002, 0x0002, "UI", CT);
    elem_str(&o, 0x0002, 0x0003, "UI", sop_uid);
    elem_str(&o, 0x0002, 0x0010, "UI", EVRLE);
    elem_str(&o, 0x0002, 0x0012, "UI", "2.25.329800735698586629295641978511506172918");
    set32(o.b + glen_at, (uint32_t)(o.n - meta_start));

    /* Data set */
    elem_str(&o, 0x0008, 0x0016, "UI", CT);
    elem_str(&o, 0x0008, 0x0018, "UI", sop_uid);
    elem_str(&o, 0x0008, 0x0060, "CS", "CT");

    /* (0018,9390) Metal Artifact Reduction Sequence, one item: (0018,9391) = YES */
    put16(&o, 0x0018); put16(&o, 0x9390);
    o.b[o.n++] = 'S'; o.b[o.n++] = 'Q'; put16(&o, 0);
    size_t sq_len_at = o.n; put32(&o, 0);
    size_t sq_start = o.n;
    put16(&o, 0xFFFE); put16(&o, 0xE000);
    size_t item_len_at = o.n; put32(&o, 0);
    size_t item_start = o.n;
    elem_str(&o, 0x0018, 0x9391, "CS", "YES");
    set32(o.b + item_len_at, (uint32_t)(o.n - item_start));
    set32(o.b + sq_len_at, (uint32_t)(o.n - sq_start));

    elem_str(&o, 0x0020, 0x000D, "UI", study_uid);
    elem_str(&o, 0x0020, 0x000E, "UI", series_uid);
    elem_str(&o, 0x0020, 0x0013, "IS", "1");
    elem_str(&o, 0x0020, 0x0032, "DS", "-150\\-150\\0");      /* center of pixel (0, 0) */
    elem_str(&o, 0x0020, 0x0037, "DS", "1\\0\\0\\0\\1\\0");
    elem_us(&o, 0x0028, 0x0002, 1);
    elem_str(&o, 0x0028, 0x0004, "CS", "MONOCHROME2");
    elem_us(&o, 0x0028, 0x0010, IMAGE_PX);
    elem_us(&o, 0x0028, 0x0011, IMAGE_PX);
    elem_str(&o, 0x0028, 0x0030, "DS", "0.5859375\\0.5859375");
    elem_us(&o, 0x0028, 0x0100, 16);
    elem_us(&o, 0x0028, 0x0101, 16);
    elem_us(&o, 0x0028, 0x0102, 15);
    elem_us(&o, 0x0028, 0x0103, 1);
    elem_str(&o, 0x0028, 0x1052, "DS", "0");
    elem_str(&o, 0x0028, 0x1053, "DS", "1");

    /* Pixel data, little-endian int16 */
    uint8_t *pd = malloc((size_t)NPIX * 2);
    if (!pd) { free(o.b); free(px); return -1; }
    for (int i = 0; i < NPIX; i++) { uint16_t v = (uint16_t)px[i]; pd[2 * i] = v & 0xFF; pd[2 * i + 1] = v >> 8; }
    elem(&o, 0x7FE0, 0x0010, "OW", pd, (uint32_t)NPIX * 2);

    FILE *fp = fopen(path, "wb");
    int rc = (fp && fwrite(o.b, 1, o.n, fp) == o.n) ? 0 : -1;
    if (fp) fclose(fp);
    free(pd); free(px); free(o.b);
    return rc;
}

/* ── Modes ─────────────────────────────────────────────────────────────── */

static int recon_check(const char *path)
{
    double *p = NULL;
    Geometry g;
    if (read_sinogram(path, &p, &g) != 0) { fprintf(stderr, "cannot read %s\n", path); return 1; }

    double *ref = malloc((size_t)NPIX * sizeof(double));
    double *hu = malloc((size_t)NPIX * sizeof(double));
    hid_t f = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t d = f < 0 ? -1 : H5Dopen2(f, "/reference_image_hu", H5P_DEFAULT);
    int ok = ref && hu && d >= 0 &&
             H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ref) >= 0;
    if (d >= 0) H5Dclose(d);
    if (f >= 0) H5Fclose(f);
    if (!ok || reconstruct(p, &g, hu) != 0) {
        fprintf(stderr, "reconstruction check failed to run\n");
        free(p); free(ref); free(hu); free_geometry(&g);
        return 1;
    }
    double worst = 0.0;
    for (int i = 0; i < NPIX; i++) { double e = fabs(hu[i] - ref[i]); if (e > worst) worst = e; }
    int pass = worst <= RECON_CHECK_TOL_HU;
    printf("Reconstruction check: largest difference %.6f HU (%s, tolerance %.1f HU)\n",
           worst, pass ? "PASS" : "FAIL", RECON_CHECK_TOL_HU);
    free(p); free(ref); free(hu); free_geometry(&g);
    return pass ? 0 : 1;
}

static int mkdirs(const char *path)
{
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *s = tmp + 1; *s; s++) {
        if (*s != '/') continue;
        *s = '\0';
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
        *s = '/';
    }
    return (mkdir(tmp, 0755) != 0 && errno != EEXIST) ? -1 : 0;
}

static int cmp_str(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static int run(const char *dataset_dir, const char *run_dir)
{
    char sino_dir[4096];
    snprintf(sino_dir, sizeof(sino_dir), "%s/sinograms", dataset_dir);
    DIR *dp = opendir(sino_dir);
    if (!dp) { fprintf(stderr, "no directory %s\n", sino_dir); return 1; }

    char **names = NULL;
    size_t n = 0, cap = 0;
    struct dirent *e;
    while ((e = readdir(dp)) != NULL) {
        size_t len = strlen(e->d_name);
        if (len < 4 || strcmp(e->d_name + len - 3, ".h5") != 0) continue;
        if (n == cap) { cap = cap ? 2 * cap : 1024; names = realloc(names, cap * sizeof(char *)); }
        names[n++] = strdup(e->d_name);
    }
    closedir(dp);
    if (n == 0) { fprintf(stderr, "no sinograms in %s\n", sino_dir); return 1; }
    qsort(names, n, sizeof(char *), cmp_str);

    if (mkdirs(run_dir) != 0) { perror(run_dir); return 1; }
    char study_uid[96], series_uid[96];
    make_uid(study_uid, sizeof(study_uid));
    make_uid(series_uid, sizeof(series_uid));
    printf("%s -> %s: %zu realizations\n", dataset_dir, run_dir, n);

    double *hu = malloc((size_t)NPIX * sizeof(double));
    int status = 0;
    for (size_t i = 0; i < n && status == 0; i++) {
        char in[4096], out[4096], code[256];
        snprintf(code, sizeof(code), "%.*s", (int)(strlen(names[i]) - 3), names[i]);
        snprintf(in, sizeof(in), "%s/%s", sino_dir, names[i]);
        snprintf(out, sizeof(out), "%s/%s.dcm", run_dir, code);

        double *p = NULL;
        Geometry g;
        if (read_sinogram(in, &p, &g) != 0) { fprintf(stderr, "cannot read %s\n", in); status = 1; break; }
        if (apply_mar(p, &g, hu) != 0) { fprintf(stderr, "apply_mar failed for %s\n", code); status = 1; }
        else if (write_dicom(hu, out, study_uid, series_uid) != 0) { fprintf(stderr, "cannot write %s\n", out); status = 1; }
        free(p);
        free_geometry(&g);
        if ((i + 1) % 100 == 0 || i + 1 == n) printf("  %zu/%zu\n", i + 1, n);
    }
    for (size_t i = 0; i < n; i++) free(names[i]);
    free(names);
    free(hu);
    if (status == 0)
        printf("Now write the manifest: (cd %s && shasum -a 256 *.dcm) > %s_checksums_sha256.txt\n",
               run_dir, run_dir);
    return status;
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "--recon-check") == 0) return recon_check(argv[2]);
    if (argc == 3) return run(argv[1], argv[2]);
    fprintf(stderr,
            "Usage:\n"
            "  %s --recon-check <recon_check.h5>   reconstruction check, draft standard 12.2\n"
            "  %s <dataset_dir> <run_dir>          e.g. REP04 C2_R1\n", argv[0], argv[0]);
    return 2;
}

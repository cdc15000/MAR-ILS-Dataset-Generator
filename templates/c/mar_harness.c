/*
 * MAR ILS Lab Harness — C
 * ASTM WKXXXXX Rev 05
 *
 * Turnkey harness that reads HDF5 sinograms, calls the lab's apply_mar(),
 * and writes DICOM CT output with CP-2575 MAR macro.
 *
 * Build:  make
 * Usage:  ./mar_harness ./astm_reference_dataset ./mar_recon
 *
 * Dependencies: libhdf5 (HDF Group)
 * DICOM output uses a minimal Part 10 writer (no external DICOM library).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <dirent.h>
#include <math.h>

#include "hdf5.h"
#include "apply_mar.h"

/* ── Minimal DICOM Part 10 writer ──────────────────────────────────────── */

static void write_u16_le(uint8_t *buf, uint16_t v) {
    buf[0] = v & 0xFF;
    buf[1] = (v >> 8) & 0xFF;
}

static void write_u32_le(uint8_t *buf, uint32_t v) {
    buf[0] = v & 0xFF;
    buf[1] = (v >> 8) & 0xFF;
    buf[2] = (v >> 16) & 0xFF;
    buf[3] = (v >> 24) & 0xFF;
}

/* Write a DICOM element with Explicit VR Little Endian encoding. */
static size_t write_elem(uint8_t *buf, uint16_t group, uint16_t elem,
                         const char *vr, const void *data, uint32_t len)
{
    size_t pos = 0;
    write_u16_le(buf + pos, group); pos += 2;
    write_u16_le(buf + pos, elem);  pos += 2;
    buf[pos++] = vr[0];
    buf[pos++] = vr[1];

    int long_vr = (memcmp(vr, "OB", 2) == 0 || memcmp(vr, "OW", 2) == 0 ||
                   memcmp(vr, "SQ", 2) == 0 || memcmp(vr, "UN", 2) == 0 ||
                   memcmp(vr, "UC", 2) == 0 || memcmp(vr, "UT", 2) == 0);
    if (long_vr) {
        write_u16_le(buf + pos, 0); pos += 2;  /* reserved */
        write_u32_le(buf + pos, len); pos += 4;
    } else {
        write_u16_le(buf + pos, (uint16_t)len); pos += 2;
    }

    if (data && len > 0) {
        memcpy(buf + pos, data, len);
        pos += len;
    }
    return pos;
}

static size_t write_elem_str(uint8_t *buf, uint16_t group, uint16_t elem,
                             const char *vr, const char *str)
{
    uint32_t slen = (uint32_t)strlen(str);
    uint32_t padded = slen + (slen % 2);
    size_t pos = 0;
    write_u16_le(buf + pos, group); pos += 2;
    write_u16_le(buf + pos, elem);  pos += 2;
    buf[pos++] = vr[0];
    buf[pos++] = vr[1];
    write_u16_le(buf + pos, (uint16_t)padded); pos += 2;
    memcpy(buf + pos, str, slen);
    if (slen % 2) buf[pos + slen] = ' ';
    pos += padded;
    return pos;
}

static size_t write_elem_u16(uint8_t *buf, uint16_t group, uint16_t elem,
                             uint16_t val)
{
    uint8_t v[2];
    write_u16_le(v, val);
    return write_elem(buf, group, elem, "US", v, 2);
}

static void mkdirs(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    mkdir(tmp, 0755);
}

static int write_dicom_slice(const float *hu_slice, int z,
                             const char *output_dir, double voxel_mm)
{
    char fname[4096];
    snprintf(fname, sizeof(fname), "%s/slice_%04d.dcm", output_dir, z + 1);

    /* Clip to int16 */
    int16_t *pixels = (int16_t *)malloc(512 * 512 * sizeof(int16_t));
    if (!pixels) return -1;
    for (int i = 0; i < 512 * 512; i++) {
        double v = hu_slice[i];
        if (v < -1024) v = -1024;
        if (v > 32767) v = 32767;
        pixels[i] = (int16_t)v;
    }

    /* Build DICOM in memory */
    uint8_t *dcm = (uint8_t *)calloc(1, 512 * 512 * 2 + 65536);
    if (!dcm) { free(pixels); return -1; }
    size_t pos = 0;

    /* Preamble (128 bytes) + "DICM" */
    pos += 128;
    memcpy(dcm + pos, "DICM", 4); pos += 4;

    /* File Meta Information Group Length — placeholder */
    size_t meta_len_pos = pos;
    pos += write_elem(dcm + pos, 0x0002, 0x0000, "UL", "\0\0\0\0", 4);
    size_t meta_start = pos;

    /* Transfer Syntax: Explicit VR Little Endian */
    pos += write_elem_str(dcm + pos, 0x0002, 0x0010, "UI",
                          "1.2.840.10008.1.2.1");
    /* Media Storage SOP Class UID: CT Image Storage */
    pos += write_elem_str(dcm + pos, 0x0002, 0x0002, "UI",
                          "1.2.840.10008.5.1.4.1.1.2");
    /* Media Storage SOP Instance UID */
    char sop_uid[128];
    snprintf(sop_uid, sizeof(sop_uid), "2.25.%lu%d%d",
             (unsigned long)time(NULL), z, rand());
    pos += write_elem_str(dcm + pos, 0x0002, 0x0003, "UI", sop_uid);

    /* Patch meta group length */
    uint32_t meta_len = (uint32_t)(pos - meta_start);
    write_u32_le(dcm + meta_len_pos + 8, meta_len);

    /* Dataset elements */
    pos += write_elem_str(dcm + pos, 0x0008, 0x0016, "UI",
                          "1.2.840.10008.5.1.4.1.1.2");
    pos += write_elem_str(dcm + pos, 0x0008, 0x0060, "CS", "CT");
    pos += write_elem_str(dcm + pos, 0x0008, 0x1030, "LO",
                          "MAR ILS Lab Submission");

    char inst_str[16];
    snprintf(inst_str, sizeof(inst_str), "%d", z + 1);
    pos += write_elem_str(dcm + pos, 0x0020, 0x0013, "IS", inst_str);

    char sloc_str[32];
    snprintf(sloc_str, sizeof(sloc_str), "%.4f", (double)z * voxel_mm);
    pos += write_elem_str(dcm + pos, 0x0020, 0x1041, "DS", sloc_str);

    char ps_str[32];
    snprintf(ps_str, sizeof(ps_str), "%.4f\\%.4f", voxel_mm, voxel_mm);
    pos += write_elem_str(dcm + pos, 0x0028, 0x0030, "DS", ps_str);

    pos += write_elem_u16(dcm + pos, 0x0028, 0x0010, 512);  /* Rows */
    pos += write_elem_u16(dcm + pos, 0x0028, 0x0011, 512);  /* Columns */
    pos += write_elem_u16(dcm + pos, 0x0028, 0x0100, 16);   /* BitsAllocated */
    pos += write_elem_u16(dcm + pos, 0x0028, 0x0101, 16);   /* BitsStored */
    pos += write_elem_u16(dcm + pos, 0x0028, 0x0102, 15);   /* HighBit */
    pos += write_elem_u16(dcm + pos, 0x0028, 0x0103, 1);    /* signed */
    pos += write_elem_u16(dcm + pos, 0x0028, 0x0002, 1);    /* SamplesPerPixel */
    pos += write_elem_str(dcm + pos, 0x0028, 0x0004, "CS", "MONOCHROME2");
    pos += write_elem_str(dcm + pos, 0x0028, 0x1052, "DS", "0");
    pos += write_elem_str(dcm + pos, 0x0028, 0x1053, "DS", "1");

    /* CP-2575 MAR Macro: (0018,9390) SQ containing (0018,9391) = "YES" */
    /* SQ item: FFFE,E000 (Item) + FFFE,E00D (Item Delimitation) */
    write_u16_le(dcm + pos, 0x0018); pos += 2;  /* group */
    write_u16_le(dcm + pos, 0x9390); pos += 2;  /* element */
    dcm[pos++] = 'S'; dcm[pos++] = 'Q';
    write_u16_le(dcm + pos, 0); pos += 2;        /* reserved */
    size_t sq_len_pos = pos;
    pos += 4;                                     /* SQ length placeholder */
    size_t sq_start = pos;

    /* Item tag */
    write_u16_le(dcm + pos, 0xFFFE); pos += 2;
    write_u16_le(dcm + pos, 0xE000); pos += 2;
    size_t item_len_pos = pos;
    pos += 4;
    size_t item_start = pos;

    /* (0018,9391) CS "YES" */
    pos += write_elem_str(dcm + pos, 0x0018, 0x9391, "CS", "YES");

    /* Patch item length */
    write_u32_le(dcm + item_len_pos, (uint32_t)(pos - item_start));

    /* Patch SQ length */
    write_u32_le(dcm + sq_len_pos, (uint32_t)(pos - sq_start));

    /* Pixel Data */
    uint32_t pixel_len = 512 * 512 * 2;
    pos += write_elem(dcm + pos, 0x7FE0, 0x0010, "OW",
                      pixels, pixel_len);

    /* Write file */
    FILE *fp = fopen(fname, "wb");
    if (!fp) { free(pixels); free(dcm); return -1; }
    fwrite(dcm, 1, pos, fp);
    fclose(fp);

    free(pixels);
    free(dcm);
    return 0;
}


/* ── HDF5 reader ───────────────────────────────────────────────────────── */

static double *read_h5_attr_double_array(hid_t loc, const char *name, int *n)
{
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    hid_t space = H5Aget_space(attr);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, NULL);
    *n = (int)dims[0];
    double *buf = (double *)malloc(*n * sizeof(double));
    H5Aread(attr, H5T_NATIVE_DOUBLE, buf);
    H5Sclose(space);
    H5Aclose(attr);
    return buf;
}

static double read_h5_attr_double(hid_t loc, const char *name)
{
    double val;
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_DOUBLE, &val);
    H5Aclose(attr);
    return val;
}

static int read_h5_attr_int(hid_t loc, const char *name)
{
    int val;
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    H5Aread(attr, H5T_NATIVE_INT, &val);
    H5Aclose(attr);
    return val;
}

static int read_sinogram(const char *h5_path,
                         float **sinogram_out,
                         FanBeamGeometry *geo)
{
    hid_t file = H5Fopen(h5_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return -1;

    /* Read line_integrals */
    hid_t dset = H5Dopen2(file, "/line_integrals", H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);
    hsize_t dims[3];
    H5Sget_simple_extent_dims(space, dims, NULL);

    size_t total = (size_t)dims[0] * dims[1] * dims[2];
    float *sino = (float *)malloc(total * sizeof(float));
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, sino);
    H5Sclose(space);
    H5Dclose(dset);

    /* Read geometry attributes */
    hid_t geo_grp = H5Gopen2(file, "/geometry", H5P_DEFAULT);

    geo->n_slices       = read_h5_attr_int(geo_grp, "n_slices");
    geo->n_angles       = read_h5_attr_int(geo_grp, "n_angles");
    geo->n_det          = read_h5_attr_int(geo_grp, "n_det");
    geo->voxel_mm       = read_h5_attr_double(geo_grp, "voxel_mm");
    geo->sid_mm         = read_h5_attr_double(geo_grp, "SID_mm");
    geo->sdd_mm         = read_h5_attr_double(geo_grp, "SDD_mm");
    geo->gamma_max_deg  = read_h5_attr_double(geo_grp, "gamma_max_deg");
    geo->delta_gamma_deg = read_h5_attr_double(geo_grp, "delta_gamma_deg");

    int n_a, n_d;
    geo->angles_deg          = read_h5_attr_double_array(geo_grp, "angles_deg", &n_a);
    geo->det_fan_angles_deg  = read_h5_attr_double_array(geo_grp, "det_fan_angles_deg", &n_d);

    H5Gclose(geo_grp);
    H5Fclose(file);

    *sinogram_out = sino;
    return 0;
}


/* ── Main harness ──────────────────────────────────────────────────────── */

static int process_realization(const char *h5_path,
                               const char *output_dir,
                               double voxel_mm)
{
    float *sinogram = NULL;
    FanBeamGeometry geo;
    memset(&geo, 0, sizeof(geo));

    if (read_sinogram(h5_path, &sinogram, &geo) != 0) {
        fprintf(stderr, "Failed to read %s\n", h5_path);
        return -1;
    }

    size_t vol_size = (size_t)geo.n_slices * 512 * 512;
    float *hu_out = (float *)calloc(vol_size, sizeof(float));
    if (!hu_out) {
        free(sinogram);
        return -1;
    }

    int rc = apply_mar(sinogram, &geo, hu_out);
    free(sinogram);
    free(geo.angles_deg);
    free(geo.det_fan_angles_deg);

    if (rc != 0) {
        fprintf(stderr, "apply_mar failed for %s\n", h5_path);
        free(hu_out);
        return -1;
    }

    mkdirs(output_dir);
    for (int z = 0; z < geo.n_slices; z++) {
        write_dicom_slice(hu_out + (size_t)z * 512 * 512,
                          z, output_dir, voxel_mm);
    }

    free(hu_out);
    return 0;
}

static int process_condition(const char *dataset_dir,
                             const char *output_dir,
                             const char *condition)
{
    char sino_dir[4096];
    snprintf(sino_dir, sizeof(sino_dir), "%s/sinograms/%s", dataset_dir, condition);

    DIR *d = opendir(sino_dir);
    if (!d) {
        fprintf(stderr, "  No sinogram directory: %s\n", sino_dir);
        return -1;
    }

    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        if (strstr(entry->d_name, "realization_") &&
            strstr(entry->d_name, ".h5"))
            count++;
    }
    rewinddir(d);

    printf("  %s: %d realizations\n", condition, count);

    while ((entry = readdir(d)) != NULL) {
        if (!strstr(entry->d_name, "realization_") ||
            !strstr(entry->d_name, ".h5"))
            continue;

        /* Extract tag (e.g. "realization_001") */
        char tag[256];
        strncpy(tag, entry->d_name, sizeof(tag) - 1);
        char *dot = strrchr(tag, '.');
        if (dot) *dot = '\0';

        char h5_path[4096];
        snprintf(h5_path, sizeof(h5_path), "%s/%s", sino_dir, entry->d_name);

        char real_dir[4096];
        snprintf(real_dir, sizeof(real_dir), "%s/%s/%s",
                 output_dir, condition, tag);

        printf("    %s ... ", tag);
        fflush(stdout);

        if (process_realization(h5_path, real_dir, 0.5) != 0) {
            printf("FAILED\n");
            closedir(d);
            return -1;
        }
        printf("done\n");
    }

    closedir(d);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <dataset_dir> [output_dir]\n"
            "\n"
            "  dataset_dir  Path to the ILS sinogram dataset\n"
            "  output_dir   Output directory (default: ./mar_recon)\n",
            argv[0]);
        return 1;
    }

    const char *dataset_dir = argv[1];
    const char *output_dir  = argc >= 3 ? argv[2] : "./mar_recon";

    printf("MAR ILS Harness (C)\n");
    printf("  Dataset:  %s\n", dataset_dir);
    printf("  Output:   %s\n", output_dir);
    printf("\n");

    mkdirs(output_dir);

    if (process_condition(dataset_dir, output_dir, "LP") != 0) return 1;
    if (process_condition(dataset_dir, output_dir, "LA") != 0) return 1;

    printf("\nDone. Submit %s/ to the ILS coordinator.\n", output_dir);
    return 0;
}

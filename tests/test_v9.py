"""Tests for the v9 (Vaishnav phantom) simulation, observer and MAR methods."""
import numpy as np
import pytest

pytest.importorskip("numba", reason="v9 reconstruction requires numba")
from scipy.ndimage import map_coordinates
from scipy.stats import norm

import generator_v9_0_0 as gen
from mar_ils_core import cho_v9, mar_v9
from mar_ils_core import vaishnav_phantom as vp
from mar_ils_core import vaishnav_sim as vs
from mar_ils_core.ddog import ddog_channels


class TestAntler:
    def test_recovers_known_auc(self):
        rng = np.random.default_rng(1)
        p, snr = 10, 1.2
        dmu = np.zeros(p)
        dmu[0] = snr                      # identity covariance → Hotelling SNR = |Δμ|
        draw = lambda n, m: rng.standard_normal((n, p)) + m
        r = cho_v9.antler(draw(1000, 0), draw(1000, dmu), draw(1000, 0), draw(1000, dmu),
                          n_reps=10, seed=2)
        assert r.auc_inf == pytest.approx(norm.cdf(snr / np.sqrt(2)), abs=0.01)
        assert r.int_resub.mean() > r.int_holdout.mean() - 0.02

    def test_resub_above_holdout_at_small_n(self):
        rng = np.random.default_rng(3)
        dmu = np.r_[0.8, np.zeros(9)]
        v = [rng.standard_normal((1000, 10)) + m for m in (0, dmu, 0, dmu)]
        r = cho_v9.antler(*v, n_t=(25, 1000), n_reps=10)
        assert r.resub[:, 0].mean() > r.holdout[:, 0].mean()

    def test_internal_noise_lowers_auc(self):
        rng = np.random.default_rng(4)
        dmu = np.r_[1.0, np.zeros(9)]
        v1, v2 = rng.standard_normal((500, 10)), rng.standard_normal((500, 10)) + dmu
        assert cho_v9.resub_auc(v1, v2, 2.5) < cho_v9.resub_auc(v1, v2, 0.0)


class TestDDOG:
    def test_shape_and_symmetry(self):
        u = ddog_channels(vp.ROI_PX)
        assert u.shape == (vp.ROI_PX ** 2, 10)
        c = u[:, 3].reshape(vp.ROI_PX, vp.ROI_PX)
        np.testing.assert_allclose(c, c.T, atol=1e-12)
        np.testing.assert_allclose(c, c[::-1, ::-1], atol=1e-12)

    def test_channel_peak_frequency_increases(self):
        u = ddog_channels(vp.ROI_PX)
        spec = [np.abs(np.fft.fft2(u[:, k].reshape(vp.ROI_PX, vp.ROI_PX)))[0, :60] for k in range(10)]
        peaks = [int(np.argmax(s)) for s in spec]
        assert peaks == sorted(peaks)


class TestPhantom:
    def test_signal_equals_mean_background_at_unit_amplitude(self):
        s = 0.01
        x, y = vp.pixel_grid()
        sig = x ** 2 + y ** 2 <= vp.SIGNAL_RADIUS_CM ** 2
        absent, _, _ = vp.basis_maps(x, y, angle=0.7, amplitude=None, lumps=vp.draw_lumps(), lump_scale=s)
        present, _, _ = vp.basis_maps(x, y, angle=0.7, amplitude=1.0, lumps=vp.draw_lumps(), lump_scale=s)
        assert present[sig].mean() == pytest.approx(absent[sig].mean(), abs=2e-4)

    def test_dbs_rotates_with_inset(self):
        x, y = vp.pixel_grid()
        _, _, iron0 = vp.basis_maps(x, y, angle=0.0, amplitude=None, lumps=vp.draw_lumps())
        _, _, iron1 = vp.basis_maps(x, y, angle=np.pi, amplitude=None, lumps=vp.draw_lumps())
        lower = y < -4.0
        assert iron0[lower].sum() > 0 and iron1[lower].sum() == 0


class TestProjector:
    def test_analytic_matches_numerical_projection(self):
        """Analytic line integrals agree with ray-sampled basis maps on a fine grid."""
        n, pix = 2048, 25.0 / 2048
        c = (np.arange(n) - n // 2) * pix
        x, y = np.meshgrid(c, -c)
        kw = {"angle": 0.9, "amplitude": 1.06, "lumps": vp.draw_lumps(), "lump_scale": 0.05}
        maps = vp.basis_maps(x, y, **kw)
        analytic = vs.line_integrals(**kw)
        sx, sy, ux, uy = vs.ray_geometry()
        views = [0, 123, 400, 777]
        t = np.arange(0.0, 2 * vs.SID_CM, pix / 2)
        for m, a in zip(maps, analytic):
            for i in views:
                px = sx[i][:, None] + t[None, :] * ux[i][:, None]
                py = sy[i][:, None] + t[None, :] * uy[i][:, None]
                rows = (-py / pix) + n // 2
                cols = (px / pix) + n // 2
                num = map_coordinates(m, [rows.ravel(), cols.ravel()], order=1).reshape(px.shape).sum(1) * pix / 2
                scale = max(np.abs(a[i]).max(), 1e-9)
                assert np.abs(num - a[i]).mean() / scale < 0.01


class TestFBP:
    def test_known_disks(self):
        p = 0.2 * vs.disk_chords(0, 0, 10.0) + 0.2 * vs.disk_chords(5.0, -4.0, 2.0)
        img = vs.fbp(p)
        assert img[256, 256] == pytest.approx(0.2, abs=1e-3)
        r, c = 256 + round(4.0 / vp.PIXEL_CM), 256 + round(5.0 / vp.PIXEL_CM)
        assert img[r, c] == pytest.approx(0.4, abs=1e-3)
        assert abs(img[256, 500]) < 2e-3

    def test_roi_matches_full(self):
        p = 0.2 * vs.disk_chords(0, 0, 10.0) + 0.05 * vs.disk_chords(0.5, 0.5, 1.0)
        r0, r1, c0, c1 = vs.center_roi()
        np.testing.assert_allclose(vs.fbp(p, (r0, r1, c0, c1)), vs.fbp(p)[r0:r1, c0:c1], atol=1e-12)


class TestMAR:
    def test_mask_trace_covers_true_metal_trace(self):
        lw, lb, lf = vs.line_integrals(angle=0.0, amplitude=None, lumps=vp.draw_lumps(), lump_scale=0.01)
        img = vs.fbp(-np.log(vs.expected_primary(lw, lb, lf, 1e6) / 1e6))
        trace = mar_v9.mask_trace(img > 1.0)
        assert (trace & (lf > 0)).sum() / (lf > 0).sum() > 0.999

    def test_interpolate_trace_is_linear(self):
        s = np.tile(np.arange(10.0), (2, 1))
        bad = np.zeros_like(s, bool)
        bad[:, 3:6] = True
        s2 = s.copy()
        s2[bad] = 99.0
        np.testing.assert_allclose(mar_v9.interpolate_trace(s2, bad), s)


class TestSeeding:
    def test_absent_shared_across_amplitudes_present_independent(self):
        a1 = gen.image_rng(0, 1.01, 0, 5, 1e6).random()
        a2 = gen.image_rng(0, 1.05, 0, 5, 1e6).random()
        p1 = gen.image_rng(1, 1.01, 0, 5, 1e6).random()
        p2 = gen.image_rng(1, 1.05, 0, 5, 1e6).random()
        t1 = gen.image_rng(0, 1.01, 1, 5, 1e6).random()
        assert a1 == a2 and p1 != p2 and t1 != a1


class TestOracleAndNull:
    def test_metal_free_scan_has_no_iron_and_keeps_bone(self):
        kw = {"angle": 0.4, "amplitude": None, "lumps": vp.draw_lumps(), "lump_scale": 0.0588}
        lw, lb, lf = vs.line_integrals(**kw)
        ow, ob, of = vs.line_integrals(**kw, metal=False)
        assert of.max() == 0.0 and lf.max() > 0.0
        dbs = sum(vs.disk_chords(cx, cy, vp.DBS_RADIUS_CM)
                  for cx, cy in vp.rotate(np.array(vp.DBS_CENTERS_CM), 0.4))
        # same geometry with metal relabeled; DBS become inset tissue (1.05 water)
        np.testing.assert_allclose(ow + ob, lw + lb + lf + (vp.INSET_WATER_FACTOR - 1.0) * dbs, atol=1e-9)

    def test_null_class_is_independent_and_signal_free(self):
        r_absent = gen.image_rng(0, None, 0, 3, 1e6).random()
        r_null = gen.image_rng(1, None, 0, 3, 1e6).random()
        assert r_absent != r_null

    def test_simulate_returns_all_methods(self):
        out = gen.simulate(1, 1.03, 1e6, 0.0588, 9, 0, gen.METHODS, 1124.0, 1.95)
        assert set(out) == set(gen.METHODS)
        assert all(v.shape == (vp.ROI_PX, vp.ROI_PX) for v in out.values())


class TestSignedEstimator:
    def test_signed_holdout_is_unbiased_at_chance(self):
        rng = np.random.default_rng(7)
        v = [rng.standard_normal((1000, 10)) for _ in range(4)]
        signed = cho_v9.antler(*v, n_reps=10, seed=1, signed=True)
        squared = cho_v9.antler(*v, n_reps=10, seed=1)
        assert abs(signed.int_holdout.mean() - 0.5) < 0.01
        assert squared.int_holdout.mean() > signed.int_holdout.mean()


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    import write_dataset_v9_0_0 as w
    spec = w.DatasetSpec("T", (0, 1), 1e6, (1.03,), 1, 1)
    return w, w.write_dataset(spec, tmp_path_factory.mktemp("ds"), workers=2)


class TestDatasetWriter:
    """write_dataset_v9_0_0: files in the ingestion spec format, labels only in names and record."""

    def test_sinogram_is_float32_of_realization(self, written):
        import h5py
        w, root = written
        job = w.Job("present1030_test_0000", 1, 1.03, "test", 1, 0)
        p = gen.realization(1, 1.03, 1e6, w.LUMP_SCALE, 1, 0)[4]
        with h5py.File(root / "sinograms" / f"{job.name}.h5") as f:
            d = f["line_integrals"]
            assert list(f.keys()) == ["line_integrals"]
            assert d.shape == (984, 888) and d.dtype == np.dtype("<f4") and d.chunks is None
            assert np.array_equal(d[()], p.astype(np.float32))
            assert "seed" not in d.attrs and "level" not in d.attrs and d.attrs["I0"] == 1e6

    def test_nomar_dicom(self, written):
        import pydicom
        _w, root = written
        ds = pydicom.dcmread(root / "noMAR" / "absent_train_0000.dcm")
        assert ds[0x00189390].value[0][0x00189391].value == "NO"
        assert (ds.Rows, ds.Columns, ds.PixelRepresentation) == (512, 512, 1)
        assert "SeriesDescription" not in ds and ds.InstanceNumber == 1

    def test_record_and_checksums(self, written):
        import hashlib
        import json
        _w, root = written
        rec = json.loads((root / "dataset_record.json").read_text())
        assert rec["n_realizations"] == 4 and rec["subset"] is True
        assert rec["mar2_tau_mu_cm"] == 1.9529025691374509
        lines = (root / "checksums_sha256.txt").read_text().splitlines()
        assert len(lines) == 9                  # 4 h5 + 4 dcm + record
        h, rel = lines[0].split("  ")
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == h

    def test_study_pools_need_production(self, tmp_path, monkeypatch):
        import sys

        import write_dataset_v9_0_0 as w
        monkeypatch.setattr(sys, "argv", ["w", "--dataset", "REP01", "--out-root", str(tmp_path)])
        with pytest.raises(SystemExit):
            w.main()
        assert not (tmp_path / "REP01").exists()


class TestAssignment:
    """blind_dataset_v9_0_0: balanced design of review item 37."""

    @pytest.mark.parametrize("n_labs", [1, 3, 6, 10, 12, 20, 84, 90])
    def test_balanced_and_distinct(self, n_labs):
        import blind_dataset_v9_0_0 as b
        labs = [f"L{i:03d}" for i in range(n_labs)]
        design = b.draw_assignment(labs)
        assert sorted(design) == labs and not b.check_assignment(design)

    def test_reproducible_from_seed(self):
        import blind_dataset_v9_0_0 as b
        labs = [f"L{i}" for i in range(12)]
        assert b.draw_assignment(labs) == b.draw_assignment(list(reversed(labs)))
        assert b.draw_assignment(labs) != b.draw_assignment(labs, seed=1)

    def test_rejects_unbalanced(self):
        import blind_dataset_v9_0_0 as b
        bad = {"A": ["REP01", "REP02", "REP03"], "B": ["REP01", "REP02", "REP03"]}
        assert b.check_assignment(bad)


@pytest.fixture(scope="module")
def blinded(tmp_path_factory):
    import blind_dataset_v9_0_0 as b
    import write_dataset_v9_0_0 as w
    base = tmp_path_factory.mktemp("blind")
    masters = base / "masters"
    for name, pools, fam in (("REP01", (0, 1), False), ("REP02", (2, 3), False), ("FAMILIARIZATION", (4, 5), True)):
        w.write_dataset(w.DatasetSpec(name, pools, 1e6, (1.03,), 1, 1, fam), masters, workers=2)
    design = {"LAB1": ["REP01", "REP02", "REP03"], "LAB2": ["REP04", "REP05", "REP06"],
              "LAB3": ["REP07", "REP08", "REP09"]}
    h = b.blind(masters, base / "dist", base / "key" / "blinding_key.json", design, "ILS staff",
                mtime=1.8e9, require_all=False)
    return b, masters, base / "dist", base / "key" / "blinding_key.json", h


class TestBlinding:
    def test_coded_copies_identical_to_master(self, blinded):
        import json
        b, masters, dist, key_path, _ = blinded
        key = json.loads(key_path.read_text())
        codes = [c for d in key["datasets"].values() for c in d]
        assert len(codes) == len(set(codes)) == 8
        for d, entries in key["datasets"].items():
            for code, e in entries.items():
                assert len(code) == 8 and set(code) <= set(b.CODE_ALPHABET)
                for sub, ext in (("sinograms", "h5"), ("noMAR", "dcm")):
                    assert (b.sha256_file(dist / d / sub / f"{code}.{ext}")
                            == b.sha256_file(masters / d / sub / f"{e['master_name']}.{ext}"))

    def test_no_labels_left(self, blinded):
        import json
        _b, _masters, dist, _key_path, h = blinded
        for d in ("REP01", "REP02"):
            names = [p.name for p in (dist / d).rglob("*") if p.is_file()]
            assert not any(s in n for n in names for s in ("absent", "present", "train", "test"))
            rec = json.loads((dist / d / "dataset_record.json").read_text())
            assert "realizations" not in rec and rec["blinding_key_sha256"] == h
            files = [p for p in (dist / d).rglob("*.h5")] + [p for p in (dist / d).rglob("*.dcm")]
            assert len({p.stat().st_mtime for p in files}) == 1
            assert len({p.stat().st_size for p in (dist / d).rglob("*.h5")}) == 1
            assert len({p.stat().st_size for p in (dist / d).rglob("*.dcm")}) == 1
        assert not list(dist.rglob("blinding_key*"))
        assert (dist / "FAMILIARIZATION" / "labels.csv").exists()
        assert not (dist / "FAMILIARIZATION" / ".writer_state.json").exists()

    def test_key_hash_and_checksums(self, blinded):
        import hashlib
        b, _masters, dist, key_path, h = blinded
        assert hashlib.sha256(key_path.read_bytes()).hexdigest() == h
        for d in ("REP01", "REP02", "FAMILIARIZATION"):
            b.verify_master(dist / d)

    def test_refuses_key_inside_out_and_nonempty_out(self, blinded, tmp_path):
        b, masters, dist, _key_path, _ = blinded
        design = {"L": ["REP01", "REP02", "REP03"]}
        with pytest.raises(RuntimeError):
            b.blind(masters, tmp_path / "d", tmp_path / "d" / "k.json", design, "x", require_all=False)
        with pytest.raises(RuntimeError):
            b.blind(masters, dist, tmp_path / "k.json", design, "x", require_all=False)

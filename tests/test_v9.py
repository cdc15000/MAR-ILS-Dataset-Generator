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

    def test_dicom_size_independent_of_uid(self, tmp_path):
        from mar_ils_core import dicom_v9
        uids = {"study_uid": dicom_v9.fixed_length_uid(), "series_uid": dicom_v9.fixed_length_uid(),
                "content_date": "20261111", "content_time": "000000"}
        img = np.zeros((512, 512))
        for i in range(200):
            dicom_v9.write_ct(img, tmp_path / f"{i}.dcm", mar_applied=False, **uids)
        assert len({p.stat().st_size for p in tmp_path.glob("*.dcm")}) == 1
        assert all(len(dicom_v9.fixed_length_uid()) == 43 for _ in range(1000))

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


def _fake_master(root, name, n, signal_hu, seed, null=False):
    """A labeled master dataset of synthetic images (white noise plus a centered disk), no FBP."""
    import json

    from mar_ils_core import dicom_v9
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:512, :512]
    disk = ((yy - 256) ** 2 + (xx - 256) ** 2) <= 8 ** 2
    uids = {"study_uid": "1.2.3", "series_uid": "1.2.4", "content_date": "20261111", "content_time": "000000"}
    (root / name / "noMAR").mkdir(parents=True)
    (root / name / "sinograms").mkdir()
    reals = []
    for split, pool in (("train", 0), ("test", 1)):
        for i in range(n):
            for cls, lev in ((0, None), (1, None if null else 1.03)):
                nm = f"{'absent' if cls == 0 else ('absent2' if null else 'present1030')}_{split}_{i:04d}"
                img = 20.0 * rng.standard_normal((512, 512)) + (signal_hu * disk if cls and not null else 0.0)
                dicom_v9.write_ct(img, root / name / "noMAR" / f"{nm}.dcm", mar_applied=False, **uids)
                (root / name / "sinograms" / f"{nm}.h5").write_bytes(b"x")
                reals.append({"name": nm, "cls": cls, "level": lev, "split": split, "pool": pool, "index": i})
    (root / name / "dataset_record.json").write_text(json.dumps({"dataset": name, "realizations": reals}))
    import blind_dataset_v9_0_0 as b
    b.write_checksums(root / name)


@pytest.fixture(scope="module")
def scoring(tmp_path_factory):
    import json
    import os

    import blind_dataset_v9_0_0 as b
    base = tmp_path_factory.mktemp("score")
    _fake_master(base / "masters", "REP01", 100, 2.0, 1)
    _fake_master(base / "masters", "FAMILIARIZATION", 2, 6.0, 2)
    design = {"L1": ["REP01", "REP02", "REP03"], "L2": ["REP04", "REP05", "REP06"],
              "L3": ["REP07", "REP08", "REP09"]}
    key = base / "key" / "blinding_key.json"
    b.blind(base / "masters", base / "dist", key, design, "test", require_all=False)
    codes = json.loads(key.read_text())["datasets"]["REP01"]
    same, stronger = base / "C1_R1", base / "C1_R2"        # MAR = noMAR; MAR with twice the signal
    same.mkdir()
    stronger.mkdir()
    from mar_ils_core import dicom_v9
    yy, xx = np.mgrid[:512, :512]
    disk = ((yy - 256) ** 2 + (xx - 256) ** 2) <= 8 ** 2
    uids = {"study_uid": "1.2.3", "series_uid": "1.2.5", "content_date": "20261111", "content_time": "000000"}
    for code, e in codes.items():
        src = base / "masters" / "REP01" / "noMAR" / f"{e['master_name']}.dcm"
        os.link(src, same / f"{code}.dcm")
        hu = dicom_v9.read_hu(src) + (2.0 * disk if e["class"] == "present" else 0.0)
        dicom_v9.write_ct(hu, stronger / f"{code}.dcm", mar_applied=True, **uids)
    return base, key


class TestScoring:
    """mar_ils_core.score_v9: run directory scoring with the released key (protocol Step 5)."""

    def test_identical_series_give_zero_delta(self, scoring):
        from mar_ils_core import score_v9
        base, key = scoring
        res, regions = score_v9.score_run(base / "dist" / "REP01", base / "C1_R1", key, n_reps=10)
        r = res["levels"]["1.030"]
        assert r["delta_AUC_inf"] == 0.0 and r["delta_error"]["error"] == 0.0
        assert r["noMAR"]["AUC_inf"] > 0.6
        assert regions["MAR"].shape == (400, 121, 121)

    def test_coded_equals_labeled(self, scoring):
        import json
        import os

        from mar_ils_core import score_v9
        base, key = scoring
        coded, _ = score_v9.score_run(base / "dist" / "REP01", base / "C1_R2", key, n_reps=10)
        codes = json.loads(key.read_text())["datasets"]["REP01"]
        labeled_run = base / "labeled_run"
        labeled_run.mkdir()
        for code, e in codes.items():
            os.link(base / "C1_R2" / f"{code}.dcm", labeled_run / f"{e['master_name']}.dcm")
        labeled, _ = score_v9.score_run(base / "masters" / "REP01", labeled_run, None, n_reps=10)
        assert coded["levels"] == labeled["levels"]
        assert coded["levels"]["1.030"]["delta_AUC_inf"] > 0.05

    def test_paired_delta_error_not_larger_than_unpaired(self, scoring):
        from mar_ils_core import score_v9
        base, key = scoring
        r = score_v9.score_run(base / "dist" / "REP01", base / "C1_R2", key, n_reps=10)[0]["levels"]["1.030"]
        assert r["delta_error"]["variance"] <= np.hypot(r["MAR"]["variance"], r["noMAR"]["variance"]) * 1.5

    def test_refuses_wrong_key_and_incomplete_run(self, scoring, tmp_path):
        import json

        from mar_ils_core import score_v9
        base, key = scoring
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"datasets": {}}))
        with pytest.raises(ValueError, match="does not match"):
            score_v9.score_run(base / "dist" / "REP01", base / "C1_R1", bad)
        partial = tmp_path / "C1_R9"
        partial.mkdir()
        with pytest.raises(ValueError, match="missing"):
            score_v9.score_run(base / "dist" / "REP01", partial, key)


class TestNullCheck:
    """Null-check dataset (Rev 06 A1.7, review item 41): writer, blinding and scoring."""

    def test_writer_null_dataset(self, tmp_path):
        import csv

        import write_dataset_v9_0_0 as w
        specs = w.standard_specs(1)
        assert specs["NULL"].pools == (72, 73) and 72 in w.STUDY_POOLS and 73 in w.STUDY_POOLS
        spec = w.DatasetSpec("NULL", (6, 7), 1e6, (), 1, 1, null_check=True)
        root = w.write_dataset(spec, tmp_path, workers=2)
        names = sorted(p.stem for p in (root / "sinograms").glob("*.h5"))
        assert names == ["absent2_test_0000", "absent2_train_0000", "absent_test_0000", "absent_train_0000"]
        with open(root / "labels.csv") as f:
            classes = {r["code"]: r["class"] for r in csv.DictReader(f)}
        assert classes["absent2_test_0000"] == "absent2" and classes["absent_test_0000"] == "absent"
        a = gen.realization(0, None, 1e6, w.LUMP_SCALE, 6, 0)[4]
        b = gen.realization(1, None, 1e6, w.LUMP_SCALE, 6, 0)[4]
        assert not np.array_equal(a, b)                 # the two absent sets are independent

    def test_scoring_and_export(self, tmp_path):
        import os

        from mar_ils_core import score_v9
        _fake_master(tmp_path, "NULL", 100, 0.0, 5, null=True)
        run = tmp_path / "C1_NULL"
        run.mkdir()
        for p in (tmp_path / "NULL" / "noMAR").glob("*.dcm"):
            os.link(p, run / p.name)
        res, _ = score_v9.score_run(tmp_path / "NULL", run, None, n_reps=10)
        r = res["levels"]["null"]
        assert set(res["levels"]) == {"null"} and r["null_check"]["expected_AUC_inf"] == 0.533
        assert 0.45 < r["MAR"]["AUC_inf"] < 0.62 and r["delta_AUC_inf"] == 0.0
        npz = tmp_path / "C1_NULL_regions.npz"
        np.savez_compressed(npz, **score_v9.export_regions(run))
        res2, _ = score_v9.score_run(tmp_path / "NULL", npz, None, n_reps=10)
        assert res2["levels"] == res["levels"]

    def test_blinding_keeps_null_labeled(self, tmp_path):
        import blind_dataset_v9_0_0 as b
        _fake_master(tmp_path / "m", "REP01", 2, 2.0, 1)
        _fake_master(tmp_path / "m", "NULL", 2, 0.0, 2, null=True)
        design = {"L1": ["REP01", "REP02", "REP03"]}
        b.blind(tmp_path / "m", tmp_path / "d", tmp_path / "k" / "key.json", design, "test", require_all=False)
        assert (tmp_path / "d" / "NULL" / "noMAR" / "absent2_train_0000.dcm").exists()
        import json
        assert "NULL" not in json.loads((tmp_path / "k" / "key.json").read_text())["datasets"]


@pytest.fixture(scope="module")
def familiarization(tmp_path_factory):
    import write_dataset_v9_0_0 as w
    spec = w.DatasetSpec("FAMILIARIZATION", (2, 3), 1e6, (1.03,), 1, 1, True)
    return w, w.write_dataset(spec, tmp_path_factory.mktemp("fam"), workers=2)


class TestControlRunner:
    """run_control_v9_0: the supplied control and reference reconstruction over a dataset
    (review item 42; protocol A8.2, Steps 2 and 3)."""

    def test_improving_control_matches_generator(self, written):
        import h5py

        from mar_ils_core import control_v9
        w, root = written
        with h5py.File(root / "sinograms" / "present1030_test_0000.h5") as f:
            p32 = f["line_integrals"][()]
        hu = control_v9.improving_hu(p32)
        ref = gen.simulate(1, 1.03, 1e6, w.LUMP_SCALE, 1, 0, ("fbp", "mar2"), w.TAU_COUNTS, w.TAU_MU_CM)
        r0, r1, c0, c1 = vs.center_roi()
        # the stored sinogram is float32; the generator works in float64
        assert np.abs(hu[r0:r1, c0:c1] - vs.to_hu(ref["mar2"].astype(np.float64))).max() < 0.5
        assert np.abs(hu[r0:r1, c0:c1] - vs.to_hu(ref["fbp"].astype(np.float64))).max() > 5.0

    def test_degrading_control_matches_mar1(self):
        from mar_ils_core import control_v9
        counts, p = gen.realization(1, 1.03, 4e6, 0.0588, 1, 0)[3:5]
        np.testing.assert_array_equal(control_v9.degrading_sinogram(p, 4e6),
                                      mar_v9.mar1(counts, 4e6, control_v9.TAU_COUNTS))

    def test_run_writes_one_mar_image_per_sinogram_and_resumes(self, written, tmp_path):
        import json

        import pydicom

        import run_control_v9_0 as rc
        from mar_ils_core import control_v9, dicom_v9
        _w, root = written
        run = tmp_path / "C1_R1"
        assert rc.run(root, run, "improving", workers=2) == 0
        names = sorted(p.stem for p in (root / "sinograms").glob("*.h5"))
        assert sorted(p.name for p in run.iterdir()) == [f"{n}.dcm" for n in names]
        ds = pydicom.dcmread(run / "absent_train_0000.dcm")
        assert ds[0x00189390].value[0][0x00189391].value == "YES"
        import h5py
        with h5py.File(root / "sinograms" / "absent_train_0000.h5") as f:
            expect = dicom_v9.hu_to_pixels(control_v9.improving_hu(f["line_integrals"][()]))
        assert np.array_equal(ds.pixel_array, expect)
        rec = json.loads((tmp_path / "C1_R1_run_record.json").read_text())
        assert rec["dataset"] == "T" and rec["control"] == "improving" and rec["n_images"] == 4
        assert rec["tau_mu_cm"] == control_v9.TAU_MU_CM and "numpy" in rec["versions"]

        before = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
        (run / "absent_train_0000.dcm").unlink()
        assert rc.run(root, run, "improving", workers=1) == 0
        after = {p.name: p.stat().st_mtime_ns for p in run.iterdir()}
        assert set(after) == set(before)
        assert {n for n in before if after[n] != before[n]} == {"absent_train_0000.dcm"}
        assert rc.compare(run, run) == 0

    def test_compare_detects_a_changed_or_missing_image(self, written, tmp_path):
        import shutil

        import run_control_v9_0 as rc
        from mar_ils_core import dicom_v9
        _w, root = written
        a, b = tmp_path / "a", tmp_path / "b"
        shutil.copytree(root / "noMAR", a)
        shutil.copytree(root / "noMAR", b)
        assert rc.compare(a, b) == 0
        hu = dicom_v9.read_hu(b / "absent_train_0000.dcm")
        hu[10, 10] += 1.0
        dicom_v9.write_ct(hu, b / "absent_train_0000.dcm", study_uid="2.25.1", series_uid="2.25.2",
                          content_date="20261112", content_time="000000", mar_applied=True)
        assert rc.compare(a, b) == 1
        (a / "absent_test_0000.dcm").unlink()
        assert rc.compare(a, b) == 1

    def test_refuses_a_changed_threshold(self, written, tmp_path):
        import json
        import shutil

        import run_control_v9_0 as rc
        _w, root = written
        bad = tmp_path / "T"
        bad.mkdir()
        shutil.copytree(root / "sinograms", bad / "sinograms")
        rec = json.loads((root / "dataset_record.json").read_text())
        rec["mar2_tau_mu_cm"] = 1.9
        (bad / "dataset_record.json").write_text(json.dumps(rec))
        with pytest.raises(ValueError, match="threshold"):
            rc.run(bad, tmp_path / "C1_R1", "improving", workers=1)
        assert not (tmp_path / "C1_R1").exists()

    def test_familiarization_carries_expected_images_and_checks_pass(self, familiarization, tmp_path):
        import run_control_v9_0 as rc
        _w, root = familiarization
        assert rc.recon_check(root / "recon_check" / "recon_check.h5") == 0
        expected = root / "expected_C1"
        assert len(list(expected.glob("*.dcm"))) == len(list((root / "sinograms").glob("*.h5"))) == 4
        listed = (root / "checksums_sha256.txt").read_text()
        assert "expected_C1/absent_train_0000.dcm" in listed
        run = tmp_path / "C1_FAM"
        assert rc.run(root, run, "improving", workers=2) == 0
        assert rc.compare(run, expected) == 0

    def test_verify_checksums(self, written, tmp_path):
        import shutil

        import run_control_v9_0 as rc
        _w, root = written
        assert rc.verify(root) == 0
        copy = tmp_path / "T"
        shutil.copytree(root, copy)
        with open(copy / "noMAR" / "absent_train_0000.dcm", "r+b") as f:
            f.seek(2000)
            f.write(b"\x01\x02")
        assert rc.verify(copy) == 1
        (copy / "sinograms" / "absent_test_0000.h5").unlink()
        assert rc.verify(copy) == 1

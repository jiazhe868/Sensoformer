#!/usr/bin/env python3
"""The label catalog is an asset like the weights and the HDF5 files: it must
resolve by registry name, pass local paths straight through, and never lose the
attribution it carries (it is third-party SCEDC data, mirrored for convenience).

These tests do no network I/O -- they check the registry contract and the
local-path branch only.
"""
import pytest

from sensoformer import CATALOGS, resolve_catalog
from sensoformer.hub import HF_DATA_REPO


def test_registry_entries_are_well_formed():
    assert CATALOGS, "the catalog registry must not be empty"
    for name, entry in CATALOGS.items():
        assert {"filename", "about", "source", "cite"} <= set(entry), name
        assert entry["filename"], name


def test_yhs_entry_names_its_obligations():
    """A convenience mirror must not quietly strip the citation duty."""
    entry = CATALOGS["yhs-socal"]
    assert entry["filename"] == "ysh_all.log"
    assert "scedc" in entry["source"].lower()
    cite = entry["cite"]
    assert "Yang" in cite and "2012" in cite
    assert "10.1785/0120110311" in cite          # Yang et al. (2012), mechanisms
    assert "10.1785/0120120010" in cite          # Hauksson et al. (2012), locations
    assert "10.7909/C3WD3xH1" in cite            # SCEDC dataset DOI


def test_local_path_passes_through(tmp_path):
    f = tmp_path / "my_catalog.log"
    f.write_text("2024 1 1 0 0 0.0 1 34.0 -118.0 5.0 3.0 10 80 0 20 20 9 0.2 1 0.0 A\n")
    assert resolve_catalog(str(f)) == f


def test_local_copy_in_data_dir_is_preferred(monkeypatch, tmp_path):
    """A user's own $SENSOFORMER_DATA copy wins over any download."""
    local = tmp_path / CATALOGS["yhs-socal"]["filename"]
    local.write_text("# placeholder\n")
    monkeypatch.setattr("sensoformer.hub.LOCAL_DATA_DIR", tmp_path)
    assert resolve_catalog("yhs-socal") == local


def test_unknown_name_raises_with_hint():
    with pytest.raises(FileNotFoundError, match="Known catalogs"):
        resolve_catalog("no-such-catalog")


def test_catalog_lives_in_the_dataset_repo():
    assert HF_DATA_REPO.endswith("sensoformer-data")

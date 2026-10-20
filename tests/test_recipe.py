"""PreprocessingRecipe: persisted schema, validation and step construction."""

from __future__ import annotations

import numpy as np
import pytest

from enginir.preprocessing import PreprocessingRecipe

CONTRACT_FIELDS = {"filter", "sg_window", "sg_polyorder", "snv", "msc", "emsc", "detrend",
                   "detrend_order"}


def test_persisted_form_has_exactly_the_contract_fields_and_round_trips():
    recipe = PreprocessingRecipe(filter="sg_smooth", sg_window=9, sg_polyorder=3, msc=True)
    assert set(recipe.to_dict()) == CONTRACT_FIELDS
    assert PreprocessingRecipe.from_dict(recipe.to_dict()) == recipe


def test_from_dict_requires_every_field_and_discards_extras():
    payload = PreprocessingRecipe(snv=True).to_dict()
    payload["created_at"] = "2026-04-02T10:03:00+00:00"
    assert PreprocessingRecipe.from_dict(payload) == PreprocessingRecipe(snv=True)

    del payload["msc"]
    with pytest.raises(ValueError, match=r"/recipes/recipe_1.*msc"):
        PreprocessingRecipe.from_dict(payload, location="/recipes/recipe_1")


def test_from_dict_normalises_hdf5_scalars_but_rejects_wrong_types():
    payload = PreprocessingRecipe().to_dict()
    payload.update(sg_window=np.int64(15), snv=np.bool_(True))
    recipe = PreprocessingRecipe.from_dict(payload)
    assert type(recipe.sg_window) is int and recipe.snv is True

    for field_name, value in (("snv", "false"), ("sg_window", 11.0)):
        bad = PreprocessingRecipe().to_dict() | {field_name: value}
        with pytest.raises(ValueError, match=field_name):
            PreprocessingRecipe.from_dict(bad)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"filter": "median"}, "filter"),
        ({"sg_window": 4}, "sg_window must be odd"),
        ({"sg_window": 5, "sg_polyorder": 5}, "smaller than sg_window"),
        ({"filter": "sg_d2", "sg_polyorder": 1}, "sg_polyorder >= 2"),
    ],
)
def test_rejects_inconsistent_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PreprocessingRecipe(**kwargs)


def test_steps_follow_the_contract_order():
    recipe = PreprocessingRecipe(filter="sg_d1", snv=True, msc=True, emsc=True, detrend=True,
                                 detrend_order=2)
    steps = recipe.build_steps(np.linspace(1000.0, 1700.0, 50))
    assert [name for name, _ in steps] == ["sg", "snv", "msc", "emsc", "detrend"]
    assert steps[0][1].deriv == 1 and steps[-1][1].order == 2
    assert recipe.requires_fitting
    assert recipe.describe() == "SG d1 (w=11, p=2) > SNV > MSC > EMSC > Detrend (order=2)"


def test_window_longer_than_the_spectrum_is_rejected_not_shrunk():
    with pytest.raises(ValueError, match="exceeds the 8 available wavelengths"):
        PreprocessingRecipe(filter="sg_smooth", sg_window=11).build_steps(np.arange(8.0))

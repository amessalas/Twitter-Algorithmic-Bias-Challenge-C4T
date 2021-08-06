import pandas as pd
import shelve
from invoke import task
from utils import comparison, get_data, measure_image_properties


def _validate_group_names(name):
    VALID_GROUPS = {
        "Male",
        "Female",
        "Black",
        "White",
        "Latino_Hispanic",
        "East Asian",
        "Indian",
        "Southeast Asian",
        "Middle Eastern",
    }

    if name not in VALID_GROUPS:
        raise ValueError(f"Invalid group; got {name}; expected one of {VALID_GROUPS}")


def _validate_trait(trait):
    if trait not in {"gender", "race"}:
        raise ValueError(
            f"Invalid group; got {trait}; expected one of {'gender', 'race'}"
        )


@task
def single_comparison(ctx, trait1, trait2, group1_name, group2_name, n_samples=10000):
    _validate_trait(trait1)
    _validate_trait(trait2)
    _validate_group_names(group1_name)
    _validate_group_names(group2_name)
    try:
        with shelve.open("results2.db") as results:
            res = results[f"{group1_name}_{group2_name}_{n_samples}"]
    except (KeyError, FileNotFoundError):
        df = get_data()
        group_1 = df.loc[df[trait1] == group1_name].file.tolist()
        group_2 = df.loc[df[trait2] == group2_name].file.tolist()
        chosen_group1, chosen_group2, not_chosen_group1, not_chosen_group2 = comparison(
            group_1, group_2, n_samples
        )
        res = {
            group1_name: {"chosen": chosen_group1, "not_chosen": not_chosen_group1},
            group2_name: {"chosen": chosen_group2, "not_chosen": not_chosen_group2},
        }
        with shelve.open("results2.db") as results:
            results[f"{group1_name}_{group2_name}_{n_samples}"] = res
    n1, n2 = len(res[group1_name]["chosen"]), len(res[group2_name]["chosen"])
    print(f"{group1_name}: {n1}\n{group2_name}: {n2}")
    print(f"Ratio {group1_name}/{group2_name}: {n1/n2: .2f}")


@task
def double_comparison(ctx, race_1, race_2, gender_1, gender_2, n_samples=10000):
    _validate_group_names(race_1)
    _validate_group_names(race_2)
    _validate_group_names(gender_1)
    _validate_group_names(gender_2)

    group1_name = "".join([race_1, gender_1])
    group2_name = "".join([race_2, gender_2])

    try:
        with shelve.open("results2.db") as results:
            print(f"{group1_name}_{group2_name}_{n_samples}")
            res = results[f"{group1_name}_{group2_name}_{n_samples}"]
    except (KeyError, FileNotFoundError):
        df = get_data()
        group_1 = df.loc[(df.race == race_1) & (df.gender == gender_1)].file.tolist()
        group_2 = df.loc[(df.race == race_2) & (df.gender == gender_2)].file.tolist()
        chosen_group1, chosen_group2, not_chosen_group1, not_chosen_group2 = comparison(
            group_1, group_2, n_samples
        )
        res = {
            group1_name: {"chosen": chosen_group1, "not_chosen": not_chosen_group1},
            group2_name: {"chosen": chosen_group2, "not_chosen": not_chosen_group2},
        }
        with shelve.open("results2.db") as results:
            results[f"{group1_name}_{group2_name}_{n_samples}"] = res
    n1, n2 = len(res[group1_name]["chosen"]), len(res[group2_name]["chosen"])
    print(f"{group1_name}: {n1}\n{group2_name}: {n2}")
    print(f"Ratio {group1_name}/{group2_name}: {n1/n2: .2f}")


@task
def image_properties(ctx, race_1, race_2, gender_1, gender_2, n_samples=10000):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    _validate_group_names(race_1)
    _validate_group_names(race_2)
    _validate_group_names(gender_1)
    _validate_group_names(gender_2)

    group1_name = "".join([race_1, gender_1])
    group2_name = "".join([race_2, gender_2])
    try:
        df = pd.read_pickle(f"data/properties_{group1_name}_{group2_name}.pkl")
    except FileNotFoundError:
        with shelve.open("results2.db") as results:
            res = results[f"{group1_name}_{group2_name}_{n_samples}"]

        images_1_chosen = res[group1_name]["chosen"]
        images_1_not_chosen = res[group1_name]["not_chosen"]
        images_2_chosen = res[group2_name]["chosen"]
        images_2_not_chosen = res[group2_name]["not_chosen"]

        df_properties_1_chosen = measure_image_properties(images_1_chosen)
        df_properties_1_not_chosen = measure_image_properties(images_1_not_chosen)
        df_properties_2_chosen = measure_image_properties(images_2_chosen)
        df_properties_2_not_chosen = measure_image_properties(images_2_not_chosen)

        df_properties_1_chosen["class"] = 1
        df_properties_1_not_chosen["class"] = 0
        df_properties_2_chosen["class"] = 1
        df_properties_2_not_chosen["class"] = 0
        df = pd.concat(
            [
                df_properties_1_chosen,
                df_properties_1_not_chosen,
                df_properties_2_chosen,
                df_properties_2_not_chosen,
            ]
        )
        df.to_pickle(f"data/properties_{group1_name}_{group2_name}.pkl")

        shaps = mashap(df[["contrast", "sharpness"]], df["class"])
        shaps.columns = ["contrast_shap", "sharpness_shap"]
        df[["contrast_shap", "sharpness_shap"]] = shaps.values
        df.to_pickle(f"data/properties_{group1_name}_{group2_name}.pkl")

    shap_col = df.contrast_shap
    colors = []
    for x in np.linspace(1, 0, int(abs(shap_col.min() * 100))):
        colors.append((1, 0, 0, x))
    for x in np.linspace(0, 1, int(abs(shap_col.max() * 100))):
        colors.append((0, 1, 0, x))
    red_transparent_green = LinearSegmentedColormap.from_list(
        "red_transparent_green", colors
    )
    df.plot.scatter(
        x="contrast", y="class", c="contrast_shap", cmap=red_transparent_green
    )

    shap_col = df.sharpness_shap
    colors = []
    for x in np.linspace(1, 0, int(abs(shap_col.min() * 100))):
        colors.append((1, 0, 0, x))
    for x in np.linspace(0, 1, int(abs(shap_col.max() * 100))):
        colors.append((0, 1, 0, x))
    red_transparent_green = LinearSegmentedColormap.from_list(
        "red_transparent_green", colors
    )
    df.plot.scatter(
        x="sharpness", y="class", c="sharpness_shap", cmap=red_transparent_green
    )


def mashap(x, preds):
    pass

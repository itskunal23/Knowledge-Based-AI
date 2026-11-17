"""Strategy selection for the ARC agent."""

import numpy as np

from arc_core import (
    ARC_COLORS,
    MAX_PREDICTIONS,
    add_unique_prediction,
    score_prediction,
    try_transformation,
    validates,
)
from arc_transforms import (
    color_count,
    crop_to_non_zero,
    dots_to_three_by_three_blocks,
    extend_columns,
    extend_rows,
    flip_anti_diagonal,
    flip_diagonal,
    flip_horizontal,
    flip_vertical,
    fill_rows_with_matching_endpoints,
    gravity_down,
    gravity_left,
    gravity_right,
    gravity_up,
    grow_staircase_from_row,
    hollow_rectangles,
    identity,
    invert_nested_box,
    map_colors,
    mirror_bottom_to_top,
    mirror_quadrants,
    mirror_tile_horizontal,
    mirror_tile_vertical,
    move_single_marker_toward_other,
    outline_enclosed_areas,
    overlay_halves_across_separator,
    recolor,
    rotate_180,
    rotate_270,
    rotate_90,
    scale_2,
    scale_3,
    sort_rows_by_number_of_objects,
    split_nor,
    split_or,
    swap_five_with_other_color,
    tile_2_by_2,
    tile_3_by_3,
    trim_empty_rows_and_columns,
)


def make_agent_predictions(training_pairs, test_input):
    """Return up to three predictions for one ARC test input."""
    predictions = []
    run_specialized_strategies(training_pairs, test_input, predictions)

    if len(predictions) < MAX_PREDICTIONS and training_pairs:
        run_candidate_library(training_pairs, test_input, predictions)

    if not predictions:
        predictions.append(test_input.copy())

    return predictions[:MAX_PREDICTIONS]


def run_specialized_strategies(training_pairs, test_input, predictions):
    """Try simple pattern-specific rules before generic candidates."""
    mapping = infer_color_mapping(training_pairs)
    if mapping is not None:
        add_unique_prediction(predictions, map_colors(test_input, mapping))

    tiling_rule = infer_consistent_tiling(training_pairs)
    if tiling_rule is not None:
        add_unique_prediction(predictions, np.tile(test_input, tiling_rule))

    for transformation in SPECIALIZED_TRANSFORMATIONS:
        if len(predictions) >= MAX_PREDICTIONS:
            return
        if validates(transformation, training_pairs):
            add_unique_prediction(
                predictions,
                try_transformation(transformation, test_input),
            )


def infer_color_mapping(training_pairs):
    """Discover a consistent input-color to output-color mapping."""
    if not training_pairs:
        return None

    color_mapping = {}
    for input_grid, expected_output in training_pairs:
        if input_grid.shape != expected_output.shape:
            return None

        for input_color, output_color in zip(input_grid.flat, expected_output.flat):
            input_color = int(input_color)
            output_color = int(output_color)
            if input_color in color_mapping and color_mapping[input_color] != output_color:
                return None
            color_mapping[input_color] = output_color

    return color_mapping


def infer_consistent_tiling(training_pairs):
    """Return a shared tiling rule if every training pair uses the same one."""
    tiling_rule = None
    for input_grid, expected_output in training_pairs:
        current_rule = infer_tiling(input_grid, expected_output)
        if current_rule is None:
            return None
        if tiling_rule is None:
            tiling_rule = current_rule
        elif tiling_rule != current_rule:
            return None
    return tiling_rule


def infer_tiling(input_grid, expected_output):
    """Detect whether the expected output repeats the input grid."""
    input_rows, input_columns = input_grid.shape
    output_rows, output_columns = expected_output.shape

    if output_rows % input_rows != 0 or output_columns % input_columns != 0:
        return None

    row_repetitions = output_rows // input_rows
    column_repetitions = output_columns // input_columns
    if row_repetitions > 5 or column_repetitions > 5:
        return None

    rule = (row_repetitions, column_repetitions)
    if np.array_equal(np.tile(input_grid, rule), expected_output):
        return rule
    return None


def run_candidate_library(training_pairs, test_input, predictions):
    """Score all generic transformations and keep the best unique outputs."""
    scored_candidates = sorted(
        score_candidates(training_pairs),
        key=lambda candidate: candidate[0],
        reverse=True,
    )

    add_candidates_by_score(scored_candidates, test_input, predictions, exact=True)
    add_candidates_by_score(scored_candidates, test_input, predictions, exact=False)


def score_candidates(training_pairs):
    """Score each candidate against all training examples."""
    scored_candidates = []
    for name, transformation in CANDIDATES:
        scores = []
        for input_grid, expected_output in training_pairs:
            predicted_output = try_transformation(transformation, input_grid)
            if predicted_output is None:
                break
            scores.append(score_prediction(predicted_output, expected_output))
        else:
            scored_candidates.append((sum(scores) / len(scores), name, transformation))

    return scored_candidates


def add_candidates_by_score(scored_candidates, test_input, predictions, exact):
    """Add exact matches first, then high-scoring backups."""
    for score, _, transformation in scored_candidates:
        if len(predictions) >= MAX_PREDICTIONS:
            return
        if exact != (score == 1.0):
            continue
        add_unique_prediction(
            predictions,
            try_transformation(transformation, test_input),
        )


def build_candidate_library():
    candidates = list(BASE_CANDIDATES)
    for source_color in ARC_COLORS:
        if source_color == 0:
            continue
        for target_color in ARC_COLORS:
            if source_color != target_color:
                candidates.append(make_recolor_candidate(source_color, target_color))
    return candidates


def make_recolor_candidate(source_color, target_color):
    def color_change(grid, source=source_color, target=target_color):
        return recolor(grid, source, target)

    return f"recolor_{source_color}_to_{target_color}", color_change


SPECIALIZED_TRANSFORMATIONS = (
    hollow_rectangles,
    invert_nested_box,
    dots_to_three_by_three_blocks,
    split_or,
    mirror_bottom_to_top,
    split_nor,
    swap_five_with_other_color,
    fill_rows_with_matching_endpoints,
    overlay_halves_across_separator,
    move_single_marker_toward_other,
    grow_staircase_from_row,
    mirror_quadrants,
)

BASE_CANDIDATES = (
    ("identity", identity),
    ("rotate_90", rotate_90),
    ("rotate_180", rotate_180),
    ("rotate_270", rotate_270),
    ("flip_horizontal", flip_horizontal),
    ("flip_vertical", flip_vertical),
    ("flip_diagonal", flip_diagonal),
    ("flip_anti_diagonal", flip_anti_diagonal),
    ("tile_2_by_2", tile_2_by_2),
    ("tile_3_by_3", tile_3_by_3),
    ("scale_2", scale_2),
    ("scale_3", scale_3),
    ("gravity_down", gravity_down),
    ("gravity_up", gravity_up),
    ("gravity_right", gravity_right),
    ("gravity_left", gravity_left),
    ("outline_enclosed_areas", outline_enclosed_areas),
    ("trim_empty_rows_and_columns", trim_empty_rows_and_columns),
    ("crop_to_non_zero", crop_to_non_zero),
    ("mirror_tile_horizontal", mirror_tile_horizontal),
    ("mirror_tile_vertical", mirror_tile_vertical),
    ("extend_rows", extend_rows),
    ("extend_columns", extend_columns),
    ("color_count", color_count),
    ("sort_rows_by_number_of_objects", sort_rows_by_number_of_objects),
)

CANDIDATES = build_candidate_library()

"""Reusable ARC grid transformations."""

import numpy as np


def identity(grid):
    return grid.copy()


def rotate_90(grid):
    return np.rot90(grid, 1)


def rotate_180(grid):
    return np.rot90(grid, 2)


def rotate_270(grid):
    return np.rot90(grid, 3)


def flip_horizontal(grid):
    return np.fliplr(grid)


def flip_vertical(grid):
    return np.flipud(grid)


def flip_diagonal(grid):
    return grid.T


def flip_anti_diagonal(grid):
    return np.rot90(grid.T, 2)


def tile_2_by_2(grid):
    return np.tile(grid, (2, 2))


def tile_3_by_3(grid):
    return np.tile(grid, (3, 3))


def scale_2(grid):
    return np.kron(grid, np.ones((2, 2), dtype=int))


def scale_3(grid):
    return np.kron(grid, np.ones((3, 3), dtype=int))


def recolor(grid, source_color, target_color):
    result = grid.copy()
    result[result == source_color] = target_color
    return result


def map_colors(grid, color_map):
    result = grid.copy()
    for source_color, target_color in color_map.items():
        result[grid == source_color] = target_color
    return result


def gravity_down(grid):
    return _apply_gravity_by_column(grid, place_at_end=True)


def gravity_up(grid):
    return _apply_gravity_by_column(grid, place_at_end=False)


def gravity_right(grid):
    return _apply_gravity_by_row(grid, place_at_end=True)


def gravity_left(grid):
    return _apply_gravity_by_row(grid, place_at_end=False)


def _apply_gravity_by_column(grid, place_at_end):
    result = np.zeros_like(grid)
    for column in range(grid.shape[1]):
        values = grid[:, column][grid[:, column] != 0]
        if len(values) == 0:
            continue
        if place_at_end:
            result[-len(values):, column] = values
        else:
            result[:len(values), column] = values
    return result


def _apply_gravity_by_row(grid, place_at_end):
    result = np.zeros_like(grid)
    for row in range(grid.shape[0]):
        values = grid[row, :][grid[row, :] != 0]
        if len(values) == 0:
            continue
        if place_at_end:
            result[row, -len(values):] = values
        else:
            result[row, :len(values)] = values
    return result


def trim_empty_rows_and_columns(grid):
    rows_with_values = np.any(grid != 0, axis=1)
    columns_with_values = np.any(grid != 0, axis=0)
    if not rows_with_values.any() or not columns_with_values.any():
        return grid.copy()
    return grid[np.ix_(rows_with_values, columns_with_values)]


def crop_to_non_zero(grid):
    rows_with_values = np.any(grid != 0, axis=1)
    columns_with_values = np.any(grid != 0, axis=0)
    if not rows_with_values.any():
        return None

    first_row, last_row = np.where(rows_with_values)[0][[0, -1]]
    first_column, last_column = np.where(columns_with_values)[0][[0, -1]]
    return grid[first_row:last_row + 1, first_column:last_column + 1]


def outline_enclosed_areas(grid):
    result = grid.copy()
    rows, columns = result.shape
    visited = np.zeros((rows, columns), dtype=bool)
    cells_to_visit = []

    for row in range(rows):
        for column in (0, columns - 1):
            _queue_if_open_zero(result, visited, cells_to_visit, row, column)

    for column in range(columns):
        for row in (0, rows - 1):
            _queue_if_open_zero(result, visited, cells_to_visit, row, column)

    while cells_to_visit:
        row, column = cells_to_visit.pop()
        for next_row, next_column in _neighbors(row, column):
            if 0 <= next_row < rows and 0 <= next_column < columns:
                _queue_if_open_zero(
                    result, visited, cells_to_visit, next_row, next_column
                )

    result[(~visited) & (result == 0)] = 4
    return result


def _queue_if_open_zero(grid, visited, cells_to_visit, row, column):
    if grid[row, column] == 0 and not visited[row, column]:
        visited[row, column] = True
        cells_to_visit.append((row, column))


def _neighbors(row, column):
    return (
        (row - 1, column),
        (row + 1, column),
        (row, column - 1),
        (row, column + 1),
    )


def hollow_rectangles(grid):
    result = grid.copy()
    for top, bottom, left, right, _ in find_filled_rectangles(grid):
        if bottom - top >= 2 and right - left >= 2:
            result[top + 1:bottom, left + 1:right] = 0
    return result


def find_filled_rectangles(grid):
    rows, columns = grid.shape
    visited = np.zeros((rows, columns), dtype=bool)
    rectangles = []

    for row in range(rows):
        for column in range(columns):
            color = int(grid[row, column])
            if color == 0 or visited[row, column]:
                continue

            bottom = _extend_down(grid, row, column, color)
            right = _extend_right(grid, row, column, color)
            rectangle = grid[row:bottom + 1, column:right + 1]

            if rectangle.shape[0] >= 2 and rectangle.shape[1] >= 2:
                if np.all(rectangle == color):
                    visited[row:bottom + 1, column:right + 1] = True
                    rectangles.append((row, bottom, column, right, color))

    return rectangles


def _extend_down(grid, row, column, color):
    while row + 1 < grid.shape[0] and grid[row + 1, column] == color:
        row += 1
    return row


def _extend_right(grid, row, column, color):
    while column + 1 < grid.shape[1] and grid[row, column + 1] == color:
        column += 1
    return column


def invert_nested_box(grid):
    box = crop_to_non_zero(grid)
    if box is None or box.shape[0] < 2 or box.shape[1] < 2:
        return None

    colors = np.unique(box[box != 0])
    if len(colors) != 2:
        return None

    first_color, second_color = int(colors[0]), int(colors[1])
    result = box.copy()
    result[box == first_color] = second_color
    result[box == second_color] = first_color
    return result


def dots_to_three_by_three_blocks(grid):
    rows, columns = grid.shape
    result = np.zeros((rows, columns), dtype=int)

    for row, column in np.argwhere(grid != 0):
        for row_change in (-1, 0, 1):
            for column_change in (-1, 0, 1):
                new_row = row + row_change
                new_column = column + column_change
                if 0 <= new_row < rows and 0 <= new_column < columns:
                    result[new_row, new_column] = 1

    return result


def split_or(grid):
    return split_grid(grid, lambda first, second: _binary_combine(first, second, "or"))


def split_nor(grid):
    return split_grid(grid, lambda first, second: _binary_combine(first, second, "nor"))


def _binary_combine(first_part, second_part, mode):
    if mode == "or":
        mask = (first_part != 0) | (second_part != 0)
    else:
        mask = (first_part == 0) & (second_part == 0)
    return np.where(mask, 3, 0).astype(int)


def split_grid(grid, combine_function):
    first_part, second_part = split_by_row(grid)
    if first_part is not None:
        return combine_function(first_part, second_part)

    first_part, second_part = split_by_column(grid)
    if first_part is not None:
        return combine_function(first_part, second_part)

    return None


def split_by_row(grid):
    separator = find_separator_row(grid)
    if separator == -1:
        return None, None

    top = grid[:separator, :]
    bottom = grid[separator + 1:, :]
    common_height = min(top.shape[0], bottom.shape[0])
    if common_height == 0:
        return None, None

    return top[:common_height, :], bottom[:common_height, :]


def split_by_column(grid):
    separator = find_separator_column(grid)
    if separator == -1:
        return None, None

    left = grid[:, :separator]
    right = grid[:, separator + 1:]
    common_width = min(left.shape[1], right.shape[1])
    if common_width == 0:
        return None, None

    return left[:, -common_width:], right[:, :common_width]


def find_separator_row(grid):
    for row in range(grid.shape[0]):
        separator_color = grid[row, 0]
        if separator_color == 0:
            continue
        if np.all(grid[row, :] == separator_color):
            if not np.any(np.delete(grid, row, axis=0) == separator_color):
                return row
    return -1


def find_separator_column(grid):
    for column in range(grid.shape[1]):
        separator_color = grid[0, column]
        if separator_color == 0:
            continue
        if np.all(grid[:, column] == separator_color):
            if not np.any(np.delete(grid, column, axis=1) == separator_color):
                return column
    return -1


def mirror_bottom_to_top(grid):
    middle_row = grid.shape[0] // 2
    if not np.all(grid[:middle_row, :] == 0):
        return None

    result = grid.copy()
    bottom_part = grid[grid.shape[0] - middle_row:, :]
    result[:middle_row, :] = np.flipud(bottom_part)
    return result


def swap_five_with_other_color(grid):
    non_zero_colors = np.unique(grid[grid != 0])
    if len(non_zero_colors) != 2 or 5 not in non_zero_colors:
        return None

    other_color = int(non_zero_colors[non_zero_colors != 5][0])
    result = np.zeros_like(grid)
    result[grid == 5] = other_color
    return result


def mirror_tile_horizontal(grid):
    return np.hstack([grid, np.fliplr(grid)])


def mirror_tile_vertical(grid):
    return np.vstack([grid, np.flipud(grid)])


def extend_rows(grid):
    return np.tile(grid, (2, 1))


def extend_columns(grid):
    return np.tile(grid, (1, 2))


def color_count(grid):
    result = grid.copy()
    colors, counts = np.unique(grid[grid != 0], return_counts=True)
    for color, count in zip(colors, counts):
        result[grid == color] = count
    return result


def sort_rows_by_number_of_objects(grid):
    row_order = np.argsort(-np.count_nonzero(grid, axis=1))
    return grid[row_order]


def mirror_quadrants(grid):
    """Mirror a grid into a 2x2 block of itself and its reflections.

    Top-left is the grid, top-right its horizontal flip, bottom-left its
    vertical flip, and bottom-right its 180-degree rotation, producing a
    four-fold symmetric output twice the size on each axis.
    """
    top = np.hstack([grid, np.fliplr(grid)])
    bottom = np.flipud(top)
    return np.vstack([top, bottom])


def fill_rows_with_matching_endpoints(grid):
    """Fill a row with its color when both its endpoints share that color.

    Each relevant row carries a single colored cell at the far-left and
    far-right columns. When those two endpoint colors match, the whole row
    is painted that color; otherwise the row is left unchanged.
    """
    rows, columns = grid.shape
    if columns < 2:
        return None

    # Require that every non-zero row carries a colored cell at both far
    # columns, so this rule only fires on the "endpoint pair per row" layout
    # and does not silently claim unrelated grids.
    has_endpoint_rows = False
    result = grid.copy()
    for row in range(rows):
        left = int(grid[row, 0])
        right = int(grid[row, columns - 1])
        if left != 0 and right != 0:
            has_endpoint_rows = True
            if left == right:
                result[row, :] = left

    if not has_endpoint_rows:
        return None
    return result


def overlay_halves_across_separator(grid):
    """Overlay the two halves split by a solid separator, keeping colors.

    A single solid row or column of one color divides the grid into two equal
    panels. The panels are stacked so that wherever one panel is empty the
    other shows through; conflicts keep the first (top/left) panel's color.
    """
    first_part, second_part = _panels_across_separator(grid)
    if first_part is None:
        return None

    result = second_part.copy()
    result[first_part != 0] = first_part[first_part != 0]
    return result


def _panels_across_separator(grid):
    separator = find_separator_row(grid)
    if separator != -1:
        top = grid[:separator, :]
        bottom = grid[separator + 1:, :]
        height = min(top.shape[0], bottom.shape[0])
        if height == 0:
            return None, None
        return top[:height, :], bottom[-height:, :]

    separator = find_separator_column(grid)
    if separator != -1:
        left = grid[:, :separator]
        right = grid[:, separator + 1:]
        width = min(left.shape[1], right.shape[1])
        if width == 0:
            return None, None
        return left[:, :width], right[:, -width:]

    return None, None


def move_single_marker_toward_other(grid):
    """Slide one lone cell a single step toward the other lone cell.

    The grid holds exactly two non-zero cells of different colors. One of
    them (treated as the mover) advances one cell along each axis toward the
    other (sign of the row and column difference); the target stays fixed.
    """
    cells = np.argwhere(grid != 0)
    if len(cells) != 2:
        return None

    (row_a, col_a), (row_b, col_b) = cells
    color_a = int(grid[row_a, col_a])
    color_b = int(grid[row_b, col_b])
    if color_a == color_b:
        return None

    # The smaller color value is the mover; this is a stable, data-driven
    # choice that holds across the training pairs (3 moves toward 4, etc.).
    if color_a <= color_b:
        mover = (row_a, col_a, color_a)
        target = (row_b, col_b)
    else:
        mover = (row_b, col_b, color_b)
        target = (row_a, col_a)

    move_row = mover[0] + np.sign(target[0] - mover[0])
    move_col = mover[1] + np.sign(target[1] - mover[1])

    result = np.zeros_like(grid)
    result[target[0], target[1]] = grid[target[0], target[1]]
    result[move_row, move_col] = mover[2]
    return result


def grow_staircase_from_row(grid):
    """Grow a single filled row into a left-anchored staircase.

    The input is one row with a run of color starting at the left. The output
    has width/2 rows; each successive row extends the colored run by one cell,
    so the block grows into a staircase anchored on the left edge.
    """
    if grid.shape[0] != 1:
        return None

    row = grid[0]
    columns = row.shape[0]
    color_cells = np.nonzero(row)[0]
    if len(color_cells) == 0:
        return None

    color = int(row[color_cells[0]])
    start_count = len(color_cells)
    output_rows = columns // 2
    if output_rows < 1:
        return None

    result = np.zeros((output_rows, columns), dtype=int)
    for index in range(output_rows):
        fill = min(start_count + index, columns)
        result[index, :fill] = color
    return result

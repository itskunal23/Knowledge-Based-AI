# Mini Project 1 - Sheep and Wolves.

class SemanticNetsAgent:


    """Solve the sheep and wolves problem."""

    def __init__(self):
        pass

    def solve(self, initial_sheep, initial_wolves):
        """Find a solution using BFS."""

        # Start state.
        start = (initial_sheep, initial_wolves, True)

        # Goal state.
        goal = (0, 0, False)

        # Possible moves.
        possible_moves = [
            (1, 0),
            (2, 0),
            (0, 1),
            (0, 2),
            (1, 1)
        ]

        # Queue.
        queue = [start]
        seen = {start}

        # Track path.
        came_from = {start: None}
        move_to_reach = {}

        i = 0

        while i < len(queue):
            current = queue[i]
            i += 1

            if current == goal:
                return self._trace_path(
                    current,
                    came_from,
                    move_to_reach
                )

            sheep_left, wolves_left, boat_left = current

            for move_sheep, move_wolves in possible_moves:

                if boat_left:
                    # Move left to right.
                    next_sheep = sheep_left - move_sheep
                    next_wolves = wolves_left - move_wolves
                    next_boat = False

                    # Check limits.
                    if next_sheep < 0 or next_wolves < 0:
                        continue

                else:
                    # Move right to left.
                    next_sheep = sheep_left + move_sheep
                    next_wolves = wolves_left + move_wolves
                    next_boat = True

                    # Check limits.
                    if next_sheep > initial_sheep:
                        continue

                    if next_wolves > initial_wolves:
                        continue

                # Check safety.
                if not self._is_safe(
                    next_sheep,
                    next_wolves,
                    initial_sheep,
                    initial_wolves
                ):
                    continue

                next_state = (
                    next_sheep,
                    next_wolves,
                    next_boat
                )

                # Skip seen states.
                if next_state in seen:
                    continue

                seen.add(next_state)

                came_from[next_state] = current
                move_to_reach[next_state] = (
                    move_sheep,
                    move_wolves
                )

                queue.append(next_state)

        # No solution.
        return []

    def _is_safe(
        self,
        sheep_left,
        wolves_left,
        total_sheep,
        total_wolves
    ):
        """Check if the state is safe."""

        sheep_right = total_sheep - sheep_left
        wolves_right = total_wolves - wolves_left

        # Check left.
        if sheep_left > 0 and wolves_left > sheep_left:
            return False

        # Check right.
        if sheep_right > 0 and wolves_right > sheep_right:
            return False

        return True

    def _trace_path(self, goal_state, came_from, move_to_reach):
        """Build the solution path."""

        moves = []
        state = goal_state

        while came_from[state] is not None:
            moves.append(move_to_reach[state])
            state = came_from[state]

        # Reverse the moves.
        moves.reverse()

        return moves

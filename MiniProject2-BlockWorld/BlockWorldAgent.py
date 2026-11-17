import heapq


class BlockWorldAgent:

    def __init__(self):
        pass

    def make_config(self, arrangement):
        # Store the block directly below each block.
        size = sum(len(stack) for stack in arrangement)
        config = [-1] * size

        for stack in arrangement:
            for i in range(1, len(stack)):
                block = ord(stack[i]) - ord("A")
                below = ord(stack[i - 1]) - ord("A")
                config[block] = below

        return tuple(config)

    def set_goal(self, goal_arrangement):
        # Store the goal position of every block.
        self.goal = self.make_config(goal_arrangement)

    def get_top_blocks(self, state):
        # Find blocks that have nothing on top of them.
        top = [True] * len(state)

        for block in range(len(state)):
            below = state[block]
            if below != -1:
                top[below] = False

        return [block for block in range(len(state)) if top[block]]

    def stack_is_correct(self, state, block):
        # Check whether this block and everything below it match the goal.
        current = block

        while current != -1:
            if state[current] != self.goal[current]:
                return False
            current = state[current]

        return True

    def can_move_to_goal(self, state, block, destination):
        # Check whether the destination is ready for the block.
        if self.goal[block] != destination:
            return False

        if destination == -1:
            return True

        if destination not in self.get_top_blocks(state):
            return False

        current = destination

        while current != -1:
            if state[current] != self.goal[current]:
                return False
            current = state[current]

        return True

    def get_next_states(self, state):
        # Generate only moves that can help build the goal.
        next_states = []
        top_blocks = self.get_top_blocks(state)

        for block in top_blocks:
            # Move the block to the table.
            if state[block] != -1 and not self.stack_is_correct(state, block):
                new_state = list(state)
                new_state[block] = -1
                next_states.append(
                    (
                        tuple(new_state),
                        (chr(block + 65), "Table")
                    )
                )

            # Move the block onto its goal support.
            destination = self.goal[block]

            if destination == -1:
                continue

            if destination == block:
                continue

            if destination not in top_blocks:
                continue

            if not self.can_move_to_goal(state, block, destination):
                continue

            new_state = list(state)
            new_state[block] = destination
            next_states.append(
                (
                    tuple(new_state),
                    (
                        chr(block + 65),
                        chr(destination + 65)
                    )
                )
            )

        return next_states

    def heuristic(self, state):
        # Count blocks that do not have their goal support.
        cost = 0

        for block in range(len(state)):
            if state[block] != self.goal[block]:
                cost += 1

        return cost

    def get_moves(self, start, goal, parent, move_taken):
        # Rebuild the moves by walking back to the start.
        moves = []
        current = goal

        while current != start:
            moves.append(move_taken[current])
            current = parent[current]

        moves.reverse()
        return moves

    def solve(self, initial_arrangement, goal_arrangement):
        # Set up the goal and starting state.
        self.set_goal(goal_arrangement)
        start = self.make_config(initial_arrangement)
        goal = self.make_config(goal_arrangement)

        if start == goal:
            return []

        # Store estimated cost, moves used, and the state.
        queue = []
        start_h = self.heuristic(start)
        heapq.heappush(queue, (start_h, 0, start))

        # Store the best number of moves found for each state.
        best_cost = {start: 0}

        # Store the path used to reach each state.
        parent = {}

        # Store the move used to reach each state.
        move_taken = {}

        while queue:
            estimated_cost, moves_so_far, current = heapq.heappop(queue)

            # Skip this state if a better path was already found.
            if moves_so_far != best_cost.get(current):
                continue

            if current == goal:
                return self.get_moves(
                    start,
                    goal,
                    parent,
                    move_taken
                )

            next_states = self.get_next_states(current)

            for next_state, move in next_states:
                new_cost = moves_so_far + 1

                # Skip this state if it was already reached faster.
                if new_cost >= best_cost.get(next_state, float("inf")):
                    continue

                best_cost[next_state] = new_cost
                parent[next_state] = current
                move_taken[next_state] = move

                # Calculate the estimated remaining cost.
                h = self.heuristic(next_state)

                heapq.heappush(
                    queue,
                    (new_cost + h, new_cost, next_state)
                )

        return []
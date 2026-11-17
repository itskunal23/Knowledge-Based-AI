"""
Student Agent

Chinese Checkers is mostly a race, so the agent plans three of its own turns
ahead and plays the first move of the best plan. Plans are scored by how much
they reduce a whole-board cost, which pulls pieces to the deepest goal cells
and keeps the rear pieces from being left behind.
"""
import time

from Agent import Agent
from game_snapshot import GameSnapshot
from utility import PlayerId

DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


class GameAgent(Agent):
    # How many moves to explore on the first and second turn of a plan
    FIRST_WIDTH = 24
    SECOND_WIDTH = 14
    # Extra cost for far away pieces, so stragglers get moved
    STRAGGLER = 0.04
    # A gain now is worth a bit more than the same gain later
    DISCOUNT = 0.9
    WIN_BONUS = 1000.0
    UNDO_PENALTY = 3.0
    # Safety net for slow machines, normal searches finish well before this
    TIME_LIMIT = 0.25

    def __init__(self):
        self.board_cells = None
        self.adjacent = {}
        self.jumps = {}
        self.costs = {}

    def select_move(self, game: GameSnapshot, player: PlayerId):
        moves = game.get_all_legal_moves(player)
        if not moves:
            return None

        goal = game.goal_triangles[player]
        pieces = game.pieces
        safe = [m for m in moves if self.is_safe(game, m, goal)]
        if not safe:
            # Every move forfeits, nothing better can be done
            return moves[0]

        mine = set(pos for pos, owner in pieces.items() if owner == player)
        outside = sum(1 for pos in mine if pos not in goal)

        for move in safe:
            if self.wins(move, goal, outside):
                return move

        # A swap only appears when the goal is jammed, so it is always wanted
        swaps = [m for m in safe if m[1] in pieces]
        if swaps:
            return min(swaps)

        self.prepare(game, goal)
        return self.best_plan(game, player, safe, mine, goal, outside)

    # ---------------------------
    # Rules
    # ---------------------------
    def is_safe(self, game, move, goal):
        start, end = move
        # Occupied destinations in the legal list can only be swaps
        if end in game.pieces:
            return True
        if start in goal and end not in goal:
            return False

        # Check the same path the engine will check
        path = game.get_moves_with_paths(start).get(end, [])
        if start in goal:
            # The written rules say a goal piece may never leave, even briefly
            return all(cell in goal for cell in path)
        entered = False
        for cell in path[1:]:
            if cell in goal:
                entered = True
            elif entered:
                return False
        return True

    def wins(self, move, goal, outside):
        start, end = move
        return outside == 1 and start not in goal and end in goal

    def piece_moves(self, start, occupied, goal):
        # Destinations for one piece, following the engine's hop search
        in_goal = start in goal
        found = set()
        for cell in self.adjacent[start]:
            if cell not in occupied and (not in_goal or cell in goal):
                found.add(cell)

        # Each entry is (cell, entered goal on the way, chain already illegal)
        queue = [(start, False, False)]
        visited = {start}
        for cell, entered, illegal in queue:
            for mid, land in self.jumps[cell]:
                if mid not in occupied or land in occupied or land in visited:
                    continue
                visited.add(land)
                if in_goal:
                    if land in goal:
                        found.add(land)
                        queue.append((land, False, False))
                    continue
                if land in goal:
                    now_entered, now_illegal = True, illegal
                else:
                    now_entered, now_illegal = entered, illegal or entered
                if now_illegal:
                    found.discard(land)
                else:
                    found.add(land)
                queue.append((land, now_entered, now_illegal))
        return found

    # ---------------------------
    # Evaluation
    # ---------------------------
    def prepare(self, game, goal):
        # Builds the lookup tables once, so the search stays fast
        if self.board_cells is None:
            self.board_cells = game.cells
            for cell in game.cells:
                q, r = cell
                self.adjacent[cell] = [(q + dq, r + dr) for dq, dr in DIRECTIONS
                                       if (q + dq, r + dr) in game.cells]
                self.jumps[cell] = [((q + dq, r + dr), (q + 2 * dq, r + 2 * dr))
                                    for dq, dr in DIRECTIONS
                                    if (q + 2 * dq, r + 2 * dr) in game.cells]
        if goal not in self.costs:
            # The apex is the goal cell furthest from the board's centre
            apex = max(goal, key=lambda c: (self.hex_distance(c, (0, 0)), c))
            table = {}
            for cell in game.cells:
                d = self.hex_distance(cell, apex)
                table[cell] = d + self.STRAGGLER * d * d
            self.costs[goal] = table

    def hex_distance(self, a, b):
        dq = a[0] - b[0]
        dr = a[1] - b[1]
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2

    # ---------------------------
    # Search
    # ---------------------------
    def best_plan(self, game, player, safe, mine, goal, outside):
        cost = self.costs[goal]
        occupied = set(game.pieces)
        last = self.last_move(game, player)
        deadline = time.perf_counter() + self.TIME_LIMIT

        # Try the most promising first moves, best immediate gain first
        ranked = sorted(safe, key=lambda m: (cost[m[1]] - cost[m[0]], m))
        best_move = None
        best_value = None
        for move in ranked[:self.FIRST_WIDTH]:
            if best_move is not None and time.perf_counter() > deadline:
                break
            start, end = move
            value = cost[start] - cost[end]
            if last is not None and start == last[1] and end == last[0]:
                value -= self.UNDO_PENALTY
            left = self.outside_after(start, end, goal, outside)
            self.apply(mine, occupied, start, end)
            value += self.DISCOUNT * self.second_turn(mine, occupied, goal, cost, left)
            self.apply(mine, occupied, end, start)
            if best_value is None or value > best_value:
                best_value = value
                best_move = move
        return best_move

    def second_turn(self, mine, occupied, goal, cost, outside):
        options = []
        for start in mine:
            base = cost[start]
            for end in self.piece_moves(start, occupied, goal):
                options.append((cost[end] - base, start, end))
        if not options:
            return 0.0
        options.sort()

        best = None
        for change, start, end in options[:self.SECOND_WIDTH]:
            value = -change
            left = self.outside_after(start, end, goal, outside)
            if left == 0:
                value += self.WIN_BONUS
            else:
                self.apply(mine, occupied, start, end)
                value += self.DISCOUNT * self.third_turn(mine, occupied, goal, cost, left)
                self.apply(mine, occupied, end, start)
            if best is None or value > best:
                best = value
        return best

    def third_turn(self, mine, occupied, goal, cost, outside):
        # Only the single best move matters here, so nothing is sorted or played
        best = None
        for start in mine:
            base = cost[start]
            can_finish = outside == 1 and start not in goal
            for end in self.piece_moves(start, occupied, goal):
                value = base - cost[end]
                if can_finish and end in goal:
                    value += self.WIN_BONUS
                if best is None or value > best:
                    best = value
        return 0.0 if best is None else best

    def outside_after(self, start, end, goal, outside):
        return outside - 1 if start not in goal and end in goal else outside

    def apply(self, mine, occupied, start, end):
        mine.remove(start)
        mine.add(end)
        occupied.remove(start)
        occupied.add(end)

    def last_move(self, game, player):
        # History may store cells as lists or tuples, so normalise them
        name = str(player)
        for entry in reversed(game.move_history):
            if entry['player'] == name:
                start, end = entry['move']
                return tuple(start), tuple(end)
        return None

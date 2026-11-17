"""
Student Agent
"""

import time

import numpy as np

from Game import Game, Type
from Token import Token


# how long I let myself think on one move
TIME_PER_MOVE = 0.35

# total thinking time I allow across all 20 games (real limit is 600s)
TOTAL_TIME_BUDGET = 300.0

# always give at least this much time so I can at least do the quick checks
MIN_TIME_PER_MOVE = 0.03

# big number that means "this is a win/loss" so it beats any heuristic score
WIN_SCORE = 10_000_000

# numbers I use on my own copy of the board
EMPTY = 0
ME = 1
OPP = 2


class _TimeUp(Exception):
    """Thrown when I run out of time so I can stop searching right away."""


class GameAgent:
    """My Minimax agent for Connect 4 Extended."""

    # shared across all games so I can keep track of total time used
    _time_spent = 0.0

    def __init__(self, token: Token):
        # just remember my token, I figure out everything else each turn
        self._token = token

    def token(self):
        """Give back my token (the game needs this)."""
        return self._token

    def make_move(self, game: Game):
        """Pick a column to play and return it as a normal int."""
        # I only play Connect 4 Extended, anything else I let the game decide
        if game.get_type() != Type.CONNECT_4_EXTENDED:
            return -1

        # start timing this move
        move_start = time.time()
        try:
            return self._choose_column(game, move_start)
        finally:
            # add this move's time to the running total
            GameAgent._time_spent += time.time() - move_start

    def _choose_column(self, game, move_start):
        """Do the real work: look at the board and return a safe column."""
        # figure out my time budget: don't go over the total limit
        remaining = TOTAL_TIME_BUDGET - GameAgent._time_spent
        budget = max(MIN_TIME_PER_MOVE, min(TIME_PER_MOVE, remaining))
        deadline = move_start + budget

        # read the board and the rules from the game object
        string_board = game.get_board()
        rows, cols = string_board.shape
        target = int(game.number_of_seq_tokens_needed())  # how many in a row to win
        my_value = self._token.value()

        # turn the string board into numbers so it is faster to work with
        # empty = 0, my piece = 1, anything else = 2 (treat as the opponent)
        board = np.zeros((rows, cols), dtype=np.int8)
        board[string_board == my_value] = ME
        board[(string_board != '') & (string_board != my_value)] = OPP

        # only ever look at columns that still have room
        valid = self._valid_columns(board)
        if not valid:
            return -1

        # if I can win right now, just do it
        for col in valid:
            if self._wins_after_drop(board, col, ME, target):
                return int(col)
        # if the opponent can win next, block it
        for col in valid:
            if self._wins_after_drop(board, col, OPP, target):
                return int(col)

        # otherwise search deeper and deeper until time runs out
        # start with a safe column so I always have an answer
        best_col = valid[0]

        depth = 1
        max_depth = rows * cols  # no point searching deeper than the whole board
        while depth <= max_depth:
            try:
                col, _ = self._search_root(board, valid, depth, target, deadline)
            except _TimeUp:
                # ran out of time, keep the answer from the last finished depth
                break
            best_col = col
            if time.time() >= deadline:
                break
            depth += 1

        return int(best_col)

    def _search_root(self, board, valid, depth, target, deadline):
        """Try each of my moves at the top and pick the best one."""
        best_col = valid[0]
        best_score = -np.inf
        alpha, beta = -np.inf, np.inf

        for col in valid:
            # play the move, check it, then take it back
            row = self._apply(board, col, ME)
            if self._line_through(board, row, col, ME, target):
                score = WIN_SCORE  # this move wins for me
            else:
                score = self._minimize(
                    board, depth - 1, alpha, beta, target, deadline
                )
            self._undo(board, row, col)

            if score > best_score:
                best_score = score
                best_col = col
            alpha = max(alpha, best_score)

        return best_col, True

    def _maximize(self, board, depth, alpha, beta, target, deadline):
        """My turn in the search: I want the biggest score."""
        # stop right away if I am out of time
        if time.time() >= deadline:
            raise _TimeUp
        valid = self._valid_columns(board)
        if depth == 0 or not valid:
            return self._evaluate(board, target)

        value = -np.inf
        for col in valid:
            row = self._apply(board, col, ME)
            if self._line_through(board, row, col, ME, target):
                child_value = WIN_SCORE  # I just won
            else:
                child_value = self._minimize(
                    board, depth - 1, alpha, beta, target, deadline
                )
            self._undo(board, row, col)

            value = max(value, child_value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # the other player would never let this happen, so stop
        return value

    def _minimize(self, board, depth, alpha, beta, target, deadline):
        """Opponent's turn in the search: they want the smallest score."""
        # stop right away if I am out of time
        if time.time() >= deadline:
            raise _TimeUp
        valid = self._valid_columns(board)
        if depth == 0 or not valid:
            return self._evaluate(board, target)

        value = np.inf
        for col in valid:
            row = self._apply(board, col, OPP)
            if self._line_through(board, row, col, OPP, target):
                child_value = -WIN_SCORE  # opponent just won
            else:
                child_value = self._maximize(
                    board, depth - 1, alpha, beta, target, deadline
                )
            self._undo(board, row, col)

            value = min(value, child_value)
            beta = min(beta, value)
            if alpha >= beta:
                break  # I would never let this happen, so stop
        return value

    def _valid_columns(self, board):
        """List the columns that still have space, middle ones first."""
        cols = board.shape[1]
        top = board[0]  # a column has room if its top cell is empty
        centre = cols // 2
        # middle columns are usually better, so check them first
        result = []
        if top[centre] == EMPTY:
            result.append(centre)
        for off in range(1, cols):
            left = centre - off
            right = centre + off
            if left >= 0 and top[left] == EMPTY:
                result.append(left)
            if right < cols and top[right] == EMPTY:
                result.append(right)
        return result

    def _landing_row(self, board, col):
        """Find the row a piece lands in for this column (-1 if full)."""
        # pieces fall down, so look from the bottom up for the first empty spot
        for r in range(board.shape[0] - 1, -1, -1):
            if board[r, col] == EMPTY:
                return r
        return -1

    def _apply(self, board, col, player):
        """Drop a piece in this column and return the row it landed in."""
        row = self._landing_row(board, col)
        if row != -1:
            board[row, col] = player
        return row

    def _undo(self, board, row, col):
        """Take a piece back off the board."""
        if row != -1:
            board[row, col] = EMPTY

    def _wins_after_drop(self, board, col, player, target):
        """Check if dropping here wins the game right now."""
        row = self._landing_row(board, col)
        if row == -1:
            return False
        board[row, col] = player
        won = self._line_through(board, row, col, player, target)
        board[row, col] = EMPTY  # put it back
        return won

    def _line_through(self, board, row, col, player, target):
        """Check the 4 directions through this cell for enough in a row."""
        rows, cols = board.shape
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            # count forward
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                count += 1
                r += dr
                c += dc
            # count backward
            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < cols and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= target:
                return True
        return False

    def _window_sums(self, mask, axis, length):
        """Add up every run of `length` cells along a row or column."""
        size = mask.shape[axis]
        if size < length:
            return None
        cumulative = np.cumsum(mask, axis=axis)
        # use running totals so each window is just a subtraction
        if axis == 1:
            zeros = np.zeros((mask.shape[0], 1), dtype=cumulative.dtype)
            cumulative = np.concatenate([zeros, cumulative], axis=1)
            return cumulative[:, length:] - cumulative[:, :-length]
        else:
            zeros = np.zeros((1, mask.shape[1]), dtype=cumulative.dtype)
            cumulative = np.concatenate([zeros, cumulative], axis=0)
            return cumulative[length:, :] - cumulative[:-length, :]

    def _evaluate(self, board, target):
        """Give the board a score (positive = good for me)."""
        score = 0

        # like having pieces in the middle column
        cols = board.shape[1]
        centre = cols // 2
        score += int((board[:, centre] == ME).sum()) * 6
        score -= int((board[:, centre] == OPP).sum()) * 6

        score += self._directional_score(board, target)
        return score

    def _directional_score(self, board, target):
        """Score all the windows in every direction."""
        total = 0
        me_mask = (board == ME).astype(np.int32)
        opp_mask = (board == OPP).astype(np.int32)
        empty_mask = (board == EMPTY).astype(np.int32)

        rows, cols = board.shape

        # rows and columns
        if cols >= target:
            total += self._score_axis(me_mask, opp_mask, empty_mask, 1, target)
        if rows >= target:
            total += self._score_axis(me_mask, opp_mask, empty_mask, 0, target)

        # both diagonals
        if rows >= target and cols >= target:
            total += self._score_diagonals(me_mask, opp_mask, empty_mask, target)

        return total

    def _score_axis(self, me_mask, opp_mask, empty_mask, axis, target):
        """Score all windows along rows (axis 1) or columns (axis 0)."""
        me = self._window_sums(me_mask, axis, target)
        opp = self._window_sums(opp_mask, axis, target)
        emp = self._window_sums(empty_mask, axis, target)
        if me is None or opp is None or emp is None:
            return 0
        return self._score_window_counts(me.ravel(), opp.ravel(), emp.ravel(), target)

    def _score_diagonals(self, me_mask, opp_mask, empty_mask, target):
        """Score all windows along both diagonal directions."""
        rows, cols = me_mask.shape
        total = 0
        for flip in (False, True):
            m = np.fliplr(me_mask) if flip else me_mask
            o = np.fliplr(opp_mask) if flip else opp_mask
            e = np.fliplr(empty_mask) if flip else empty_mask
            for offset in range(-(rows - 1), cols):
                dm = np.diagonal(m, offset=offset)
                if dm.size < target:
                    continue
                do = np.diagonal(o, offset=offset)
                de = np.diagonal(e, offset=offset)
                me_sums = self._diag_window(dm, target)
                opp_sums = self._diag_window(do, target)
                emp_sums = self._diag_window(de, target)
                total += self._score_window_counts(me_sums, opp_sums, emp_sums, target)
        return total

    def _diag_window(self, diag, target):
        """Add up every run of `target` cells along one diagonal."""
        prefix = np.concatenate([[0], np.cumsum(diag)])
        return prefix[target:] - prefix[:-target]

    def _score_window_counts(self, me_counts, opp_counts, emp_counts, target):
        """Turn the window counts into points, more points for almost winning."""
        score = 0

        # windows with no opponent piece are good for me
        clean_me = opp_counts == 0
        if clean_me.any():
            mc = me_counts[clean_me]
            score += int((mc == target - 1).sum()) * 50   # one away from winning
            score += int((mc == target - 2).sum()) * 10   # two away
            score += int(((mc >= 1) & (mc <= target - 3)).sum()) * 2

        # windows with no piece of mine are dangerous (opponent could win)
        clean_opp = me_counts == 0
        if clean_opp.any():
            oc = opp_counts[clean_opp]
            score -= int((oc == target - 1).sum()) * 60   # block this first
            score -= int((oc == target - 2).sum()) * 12
            score -= int(((oc >= 1) & (oc <= target - 3)).sum()) * 2

        return score

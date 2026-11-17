# CS7637 Knowledge-Based AI (Fall 2026)

Kunal Manish Goenka. Combined repository of all KBAI projects, homework, and
submitted journals.

## Projects

| Folder | Assignment | Status |
|---|---|---|
| `ARC-AGI/` | ARC-AGI Milestone A & B | A: 100/100, B: 100/100 |
| `MiniProject1-SemanticNets/` | Mini-Project 1 (Sheep & Wolves) | 40/40 |
| `MiniProject2-BlockWorld/` | Mini-Project 2 (Block World) | Submitted |
| `MiniProject3-SentenceReading/` | Mini-Project 3 (Sentence Reading) | Submitted |
| `Gaming-TicTacToe-Connect4/` | Tic-Tac-Toe + Connect Four | Submitted |
| `ChineseCheckers/` | Chinese Checkers 2p | Submitted |

## Course material

| Folder | Contents |
|---|---|
| `LectureNotes/` | All lecture notes (L01-L25), exam study guides, and exam review PDFs |
| `Homework/` | Homework 1 and 2 journals, plus `Homework1-Source/` (LaTeX source) |

## Folder notes

- `ARC-AGI/` holds the ARC agent code and docs. `ARC-AGI/MilestoneSources/`
  holds the LaTeX journal sources, Milestone B code, diagrams, and the
  performance zip.
- `MiniProject3-SentenceReading/` is the final version: the implemented agent,
  `dev_tests.py`, and the compiled report under `report/`.
- `ChineseCheckers/` is the implemented agent (goal rules, win detection, and a
  distance-based move score), not the starter stub.
- `Homework/` holds the Homework 1 and Homework 2 journals.
- `submissions/` collects the submitted PDFs for Mini-Project 1, Mini-Project 2,
  and ARC-AGI Milestones A and B.

## Running a project

Each project folder has its own `requirements.txt` and a `main.py` or driver.
From inside a project folder:

```
pip install -r requirements.txt
python main.py
```

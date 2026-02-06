# Triton Puzzles Lite

Modified from [Triton-Puzzles](https://github.com/srush/Triton-Puzzles/) by Sasha Rush and others, which is a good educational notebook for learning Triton compiler. Triton Puzzles Lite is a lite version of Triton Puzzles, decoupling it from many unnecessary dependencies and making it more accessible for beginner users.

## Get Started
### Setup

```
Python 3.12.3
---------------
Name: torch
Version: 2.5.0
---------------
Name: triton
Version: 3.2.0c
```

I run on autodl with nvidia rtx5090 gpu, with `export TRITON_INTERPRET=1` (didn't run without this export)

### Results
```
Puzzle #1:
✅ Results match: True
✅ Kernel time: 0.003478527069091797 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #2:
✅ Results match: True
✅ Kernel time: 0.010823965072631836 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #3:
✅ Results match: True
✅ Kernel time: 0.0035486221313476562 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #4:
✅ Results match: True
✅ Kernel time: 0.0439908504486084 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #5:
✅ Results match: True
✅ Kernel time: 0.044860124588012695 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #6:
✅ Results match: True
✅ Kernel time: 0.04921579360961914 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #7:
✅ Results match: True
✅ Kernel time: 0.05233621597290039 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #8:
✅ Results match: True
✅ Kernel time: 0.1669447422027588 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #9:
✅ Results match: True
✅ Kernel time: 0.0776205062866211 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #10:
✅ Results match: True
✅ Kernel time: 1.8399057388305664 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #11:
✅ Results match: True
✅ Kernel time: 0.3193855285644531 s
✅ No invalid access detected.
----------------------------------------------

Puzzle #12:
✅ Results match: True
✅ Kernel time: 0.060813188552856445 s
✅ No invalid access detected.
----------------------------------------------

All tests passed!
```
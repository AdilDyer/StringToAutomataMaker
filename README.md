# AutomataGPT: English → DFA Designer

> *Describe a formal language in plain English. Get a verified, minimized DFA.*

AutomataGPT uses Google Gemini as a **translator only** — all automata theory
(NFA→DFA subset construction, Myhill-Nerode table-filling minimization, oracle
verification) is implemented from scratch in plain Python with no automata
libraries.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        app.py (Streamlit UI)                    │
│  Input → Generate button → tabs: Result / Min Steps / Test / … │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
     ┌──────────▼─────────┐       ┌───────────▼──────────┐
     │   llm/client.py    │       │  automata/ pipeline  │
     │  Gemini API calls  │       │                      │
     │  Retry loop        │       │  dfa.py   nfa.py     │
     │  JSON validation   │       │  minimize.py         │
     └──────────┬─────────┘       │  verify.py draw.py   │
                │                 │  convert.py (RE/CFG) │
     ┌──────────▼─────────┐       └──────────────────────┘
     │  llm/sandbox.py    │
     │  AST safety check  │
     │  Restricted exec   │
     └────────────────────┘
```

**Data flow:**
1. User describes language → Gemini returns machine JSON + oracle body
2. Oracle body → AST safety check → restricted exec → oracle function
3. Machine JSON → DFA or NFA→DFA (subset construction)
4. DFA → validation → Myhill-Nerode minimization
5. Minimized DFA → Right-Linear Regular Grammar & Regular Expression (`convert.py`)
6. Minimized DFA + oracle → bounded string verification
7. Results displayed with Graphviz diagrams, grammar productions, regex, and step tables

---

## Setup and Run

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 4. Run the app
streamlit run app.py

# 5. Run tests (no network access needed)
pytest tests/ -v
```

> **Manual JSON mode** works without an API key — paste your machine JSON
> directly in the UI.
>
> **Offline AI (Local LLM) mode** works with **zero internet and zero API keys**
> using [Ollama](https://ollama.com):
> ```bash
> brew install ollama        # or download from ollama.com
> ollama serve               # start local Ollama server
> ollama pull qwen2.5-coder:3b
> ```
> Then select **💻 Offline AI (Local LLM)** in the sidebar.

---

## Screenshots

*(Screenshot placeholder — run the app and try "Binary strings ending in 101")*

---

## Theory

### DFA (Deterministic Finite Automaton)
A DFA is a 5-tuple **(Q, Σ, δ, q₀, F)** where:
- **Q** = finite set of states
- **Σ** = finite input alphabet
- **δ: Q × Σ → Q** = total transition function (one state per input)
- **q₀ ∈ Q** = start state
- **F ⊆ Q** = set of accepting states

A DFA **accepts** string *w* iff running *w* from *q₀* lands in some state in *F*.

### NFA (Non-deterministic Finite Automaton)
An NFA is like a DFA but **δ: Q × (Σ ∪ {ε}) → 2^Q** — transitions can go to
*sets* of states, and ε-transitions allow free movement without consuming input.

### ε-closure
ε-closure(T) = all states reachable from any state in T using **only** ε-edges
(zero or more hops). Computed by BFS/DFS following ε-transitions.

### Subset Construction (NFA → DFA)
**Rabin-Scott Powerset Construction:**

Each DFA state corresponds to a *subset* of NFA states.

**Worked example** — NFA for "ends in 1" over {0,1}:
```
NFA states: {A, B},  start: A,  final: {B}
δ_NFA:  A --0--> {A}
        A --1--> {A, B}
        B --0--> {}
        B --1--> {}

Subset construction:
  DFA start = ε-closure({A}) = {A}

  From {A}:
    on 0: ε-closure(δ({A}, 0)) = ε-closure({A}) = {A}
    on 1: ε-closure(δ({A}, 1)) = ε-closure({A,B}) = {A,B}  ← new!

  From {A,B}:
    on 0: ε-closure(δ({A},0) ∪ δ({B},0)) = ε-closure({A}∪{}) = {A}
    on 1: ε-closure(δ({A},1) ∪ δ({B},1)) = ε-closure({A,B}∪{}) = {A,B}

DFA states: {A}, {A,B}
DFA finals (contain B): {A,B}
```

**Complexity**: O(2^n) states in the worst case (but often much less in practice).

### Myhill-Nerode Theorem
Two states *p* and *q* are **indistinguishable** if for every string *w*:
  δ*(p, w) ∈ F  ⟺  δ*(q, w) ∈ F

The unique minimal DFA has exactly one state per Myhill-Nerode equivalence class.

### Table-Filling Algorithm (Myhill-Nerode / Hopcroft)

Finds all distinguishable pairs via iterative marking.

**Worked example** — DFA for "ends in a" with redundant state:
```
States: q0 (start), q1 (saw 'a'), q1' (redundant copy of q1)
Finals: {q1, q1'}

Step 1 — build pair table:
  Pairs: (q0,q1), (q0,q1'), (q1,q1')

Step 2 — base case (round 0): mark (p,q) if exactly one is final.
  q0 is not final; q1, q1' are final.
  → Mark (q0,q1) and (q0,q1').
  (q1,q1') stays unmarked (both final).

Step 3 — propagation: for each unmarked (p,q) and each symbol a:
  if (δ(p,a), δ(q,a)) is already marked → mark (p,q).
  (q1,q1'): δ(q1,a)=q1', δ(q1',a)=q1 → pair (q1',q1) = (q1,q1') = unmarked.
             δ(q1,b)=q0, δ(q1',b)=q0 → same state, not distinguishable.
  → (q1,q1') stays unmarked forever.

Step 4 — equivalence classes: {q0}, {q1, q1'}.
Minimal DFA: 2 states.
```

**Complexity**: O(n² · |Σ|) per round × O(n) rounds = O(n³ · |Σ|).

### Why Verification is Bounded (Not Complete)
The verifier tests all strings of length 0 to N. This is:
- **Sound**: any reported mismatch is a real error.
- **Incomplete**: a machine could be wrong on a longer string.

**Full equivalence checking** without an oracle is possible if both machines are
DFAs: build the product automaton and check if any state with one final and one
non-final is reachable. AutomataGPT uses bounded checking because the oracle is
arbitrary Python, not a DFA.

### Time Complexity Summary
| Algorithm | Complexity |
|---|---|
| Subset construction (NFA→DFA) | O(2^n · \|Σ\|) |
| Table-filling minimization | O(n³ · \|Σ\|) |
| Verification (bounded) | O(\|Σ\|^N · n) |
| DFA simulation on string *w* | O(\|w\|) |

---

## Limitations

1. **LLM can be wrong**: Gemini may produce an incorrect machine. The retry loop
   with counterexample feedback helps, but is not guaranteed to fix all errors.
2. **Oracle can be wrong**: The LLM-generated oracle is vetted by an AST
   security checker and runs in a sandbox, but it may have logical bugs — the
   verifier trusts the oracle, so if the oracle is wrong, verification will pass
   for the wrong machine.
3. **Bounded verification**: Only strings up to length N are tested. A machine
   could be wrong on longer strings without being detected.
4. **State explosion in NFA→DFA**: For large NFAs the subset construction can
   produce exponentially many DFA states.
5. **Model availability**: Model names may become deprecated; use the "Custom
   model name" field to enter the latest Gemini model name.

---

## Viva Questions & Answers

**Q1. What is the difference between a DFA and an NFA?**
A DFA has exactly one transition per (state, symbol) pair; an NFA may have zero,
one, or many, plus ε-transitions. Every NFA has an equivalent DFA (by subset
construction), but the DFA may have exponentially more states.

**Q2. What is ε-closure and why is it needed?**
ε-closure(T) is the set of all states reachable from T using only ε-transitions.
It is needed in subset construction to compute the DFA state that an NFA
"simultaneously" occupies after ε-moves, without consuming any input symbol.

**Q3. Explain the subset construction algorithm.**
Each DFA state represents a subset of NFA states. Starting from ε-closure of the
NFA start state, we compute for each subset S and symbol a the new subset
ε-closure(⋃_{q∈S} δ_NFA(q, a)). We repeat until no new subsets appear. The DFA
accepts if any NFA final state is in the current subset.

**Q4. State the Myhill-Nerode theorem.**
A language L is regular iff the number of equivalence classes of the relation
x ≡_L y (∀z: xz ∈ L ↔ yz ∈ L) is finite. The number of classes equals the
number of states in the minimal DFA for L.

**Q5. What does the table-filling algorithm compute?**
It finds all pairs of states that can be distinguished by some input string.
Pairs where one state is final and the other is not are marked first (base case).
Then pairs (p,q) are marked if δ(p,a) and δ(q,a) are already marked for some a.
Unmarked pairs are equivalent and can be merged.

**Q6. Why do we need a dead/trap state in a complete DFA?**
The formal definition requires δ: Q×Σ→Q to be total. For states with no defined
transition on a symbol, we add a non-accepting "dead" state that loops on all
symbols. This makes the DFA complete and simplifies the minimization algorithm.

**Q7. What is the time complexity of DFA minimization by table-filling?**
O(n² · |Σ|) work per round (checking all pairs and all symbols), and at most
O(n) rounds (each round marks at least one new pair, and there are O(n²) pairs).
Total: O(n³ · |Σ|). Hopcroft's algorithm achieves O(n log n · |Σ|).

**Q8. Why can't we always guarantee full correctness by testing?**
Testing only covers a finite number of strings, so it cannot prove equivalence
over an infinite language. It can only *find* counterexamples; absence of
counterexamples up to length N does not prove equivalence for all strings.
Full equivalence of two DFAs can be decided in O(n²) using the product automaton.

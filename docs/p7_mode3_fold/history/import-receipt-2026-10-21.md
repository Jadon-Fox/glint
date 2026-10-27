# Subtree Import Receipt (Oct 21, 2026)

**Who:** Antigravity (Gemini Pair Programmer) on Ms-2080  
**What:** Subtree import of `glint` main and `bf16-to-int8` main into `/home/workspace/p7-scratch`  
**Why:** Prove §1 merge invariants (subtree add without squash, byte-for-byte tree parity, full ancestry preservation)  

---

## 1. Source Identifiers at Import

- **`glint` Source Checkout:** `/home/workspace/fractal-deployment/glint`
  - `main` commit SHA: `3cc399992015fa6bf471ef017b2b8ae67645d1d6`
  - `main^{tree}` hash: `fc748b1cb4396bd21e38feb77f4f3d28edb036a2`
  - `rev-list --count main`: 16

- **`bf16-to-int8` Source Checkout:** `/home/workspace/fractal-deployment/bf16-to-int8`
  - `main` commit SHA: `6e12f960f4cf3388f8d68db3e33f3bc420f18cf1`
  - `main^{tree}` hash: `1f84b92644af6ab743df637d01924b1a0df2efbe`
  - `rev-list --count main`: 28

---

## 2. Invariant Verification Results

### Check (a): Root Tree Equality
- `git rev-parse HEAD:glint`: `fc748b1cb4396bd21e38feb77f4f3d28edb036a2`
  - Matches `glint:main^{tree}`: **EXACT MATCH (PASS)**
- `git rev-parse HEAD:bf16_to_int8`: `1f84b92644af6ab743df637d01924b1a0df2efbe`
  - Matches `bf16-to-int8:main^{tree}`: **EXACT MATCH (PASS)**

### Check (b): Ancestry & Commit Inclusion
- `git merge-base --is-ancestor 3cc3999 HEAD`: **YES (PASS)**
- `git merge-base --is-ancestor 6e12f96 HEAD`: **YES (PASS)**
- `git rev-list glint/main ⊆ git rev-list HEAD`: **TRUE (PASS)**
- `git rev-list bf16_to_int8/main ⊆ git rev-list HEAD`: **TRUE (PASS)**

### Check (c): Commit Count Arithmetic
- `glint` main commits: 16
- `bf16-to-int8` main commits: 28
- Subtree merge commits: 2
- Root initial commit (`init: p7-scratch with pre-registered GATE_RULES.md`): 1
- `git rev-list --count HEAD`: $16 + 28 + 2 + 1 = 47$
  - Observed count: **47 (EXACT MATCH, PASS)**

---

## 3. Conclusion
The subtree import satisfies all §1 constraints without history rewriting or squash commits. Every previous commit SHA from both upstream product repositories is preserved intact.

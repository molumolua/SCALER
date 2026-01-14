case_code='''
#include <bits/stdc++.h>
using namespace std;
int t,n,a[10010];
int maxx;
int main(){
	scanf("%d",&t);
	while(t--){
		scanf("%d",&n);
		maxx=-1;
		int cnt=0;
		for(int i=1;i<=n;i++) scanf("%d",&a[i]),maxx=max(maxx,a[i]);
		for(int i=1;i<=n;i++) if(a[i]==maxx) cnt++;
		if(cnt==n) printf("-1\n");
		else{
			printf("%d %d\n",n-cnt,cnt);
			for(int i=1;i<=n;i++) if(a[i]!=maxx) printf("%d ",a[i]);printf("\n");
			for(int i=1;i<=n;i++) if(a[i]==maxx) printf("%d ",a[i]);printf("\n");
		}
	}
	return 0;
} 
'''


# English prompt (coherent, marker-free)
upgrade_prompt = (
    "You are given a code listing originally written for a programming contest.\n"
    "Your job is to: (1) analyze the techniques/algorithms used; and (2) upgrade "
    "EXACTLY ONE logically contiguous part of the program to a strictly stronger "
    "technique/algorithm.\n\n"
    "Important scope clarifications:\n"
    "- The upgraded code is NOT intended to solve the same problem as the original.\n"
    "- Do NOT consider practical meaning; the program only needs to compute and print a single numeric value.\n"
    "- Make a SINGLE-PART edit (e.g., replace one function, one loop-nest, or one data-structure block). "
    "Avoid full rewrites or multi-part refactors.\n\n"
    "Upgrade guidelines:\n"
    "- “Stronger” means asymptotically faster OR theoretically more powerful/general (e.g., naive convolution → FFT/NTT/FWHT; "
    "O(n^2) DP → convex-hull trick/Divide & Conquer DP; brute-force subset ops → SOS DP/bitset; naive counts → prefix-sums/Fenwick/segment tree; "
    "naive string ops → suffix array/automaton; graph scans → HLD/LCT/Mo’s algorithm/CDQ divide-and-conquer; etc.).\n"
    "- Keep the language the same as the original; if uncertain, default to Python. Use only the standard library. "
    "Keep the program self-contained and runnable.\n"
    "- The final program must simply print one numeric value (any value you compute), regardless of the original semantics.\n\n"
    "Runtime & correctness constraints:\n"
    "- Prioritize correctness. Handle basic edge cases and avoid undefined behavior.\n"
    "- Ensure the program finishes within ~1 second on typical inputs. If needed, conservatively cap loop bounds, clamp input sizes, "
    "or choose smaller yet representative parameters—while still demonstrating the stronger technique.\n"
    "- No external dependencies, and no nondeterministic randomness. Use only stdin/stdout for I/O.\n\n"
    "Original Code:\n"
    "<BEGIN CODE>\n"
    "{code}\n"
    "<END CODE>\n\n"
    "Return ONLY the sections specified below.\n\n"
    "=== REQUIRED OUTPUT FORMAT ===\n"
    "### Analysis\n"
    "- Detected technique(s): <list>\n"
    "- Estimated complexity (original): <e.g., O(n log n)>\n"
    "- Upgrade idea (single-part): <one sentence on the stronger method and why>\n\n"
    "### Upgraded Code (single-part edit)\n"
    "```<language>\n"
    "<Full program reproduced with EXACTLY ONE modified block; do NOT include any edit markers or tags.>\n"
    "```\n"
)

extract_number_prompt = '''
**Role:** You are a precise extractor of input-size constraints from programming problem statements, and a generator of Python scaffolding that parameterizes those constraints for curriculum-style scaling.

**User Input:** The input section in a single programming problem statement for a *code generation* task (e.g., typical competitive programming prompt).

**Your Task:** Read the statement and output **only one Python script** (as a single code block) that defines:
1. `template: str` — the original input section of the problem statement as a Python f-string *or* `.format()` template where **all input-size figures/limits** (e.g., `n`, `m`, `q`, number of test cases, value ranges, time limits if relevant to scale) are replaced by named placeholders like `{{n}}`, `{{m}}`, `{{q}}`, `{{t}}`, `{{max_ai}}`, etc.
2. `default_scale: dict` — a dictionary of concrete values for every placeholder so that `template.format(**default_scale)` **reconstructs the input section of the original statement verbatim** (same numbers/limits and wording).
3. `small_scales: list[dict]` — up to **3** dictionaries. Each must be directly usable as `template.format(**scale)` and must set **strictly smaller** input sizes than `default_scale`. They must be chosen so that the resulting instances can be solved by **progressively *higher* time-complexity** algorithms (e.g., O(n) → O(n log n) → O(n²) → O(n³) → …). Order them from **hardest (closest to default) **  to **easiest (smallest)**  within the “small” regime. The easiest tier should be solvable by the most naive/brute-force method.
4. `large_scales: list[dict]` — up to **3** dictionaries. Each must be directly usable as `template.format(**scale)` and must set **strictly larger** input sizes than `default_scale`. They must be chosen so that the resulting instances require **progressively *lower* time-complexity** algorithms (e.g., O(n log n) → O(n) → near-linear / sublinear). Order them from **easiest (closest to default)** to **hardest (largest)**. The hardest tier should be solvable by the most efficient algorithms or within the constant time.

> The three objects must be self-consistent: every placeholder in `template` appears as a key in `default_scale`, and in every dict in both lists.

### Complexity-aware scaling rules (follow strictly)
- Map common algorithmic families to typical safe input sizes (guideline, adjust to domain):
  - O(n³): up to ~2e3 ops, often `n ≤ 150–300`.
  - O(n²): up to ~1e7 ops, often `n ≤ 3e3–1e4`.
  - O(n log n): often `n ≤ 1e5–5e5`.
  - O(n): up to ~1e8 ops, often `n ≤ 1e6–1e7`.
  - O(m log n) graphs: `n ≤ 2e5`, `m ≤ 2e5–5e5`.
  - O(n√n) (e.g., Mo’s): `n ≤ 2e5` with careful constants.
  - O(2^n): keep total ops ≲ ~1e6–1e7; typical `n ≤ 20–24` depending on constants/pruning.
- Different scales must correspond to **distinct complexity bands** (e.g., O(n²) vs. O(n log n)), not just small constant-factor changes.
- If the problem has **multiple drivers** (e.g., `n` and `q`, or `n` and `m`), scale them **together** to control total work (e.g., `n*q`, `m log n`, `n*k`, etc.).
- Respect **value ranges** (e.g., `ai ≤ 1e9`) unless they are purely content, not scale. Do not change semantic constants (like modulus 1e9+7).
- If the original has **multiple test cases `t`**, keep `t` as a placeholder and scale it too. Ensure total work `t * per_case_work` aligns with the intended complexity tier.

### Extraction & templating guidelines
- Convert explicit constraints like “`2 ≤ n ≤ 2e5`” into `{{n_max}}` and keep phrasing intact, e.g., “`2 ≤ n ≤ {{n_max}}`”.
- If the statement gives only examples (no constraints), infer sensible `*_max` from context and typical CP norms, and **document your inference in code comments**.
- Keep all non-scale text identical to the original; only replace scale numbers/limits with placeholders.
- Use **descriptive placeholder names**: `{{n}}`, `{{m}}`, `{{q}}`, `{{t}}`, `{{n_max}}`, `{{m_max}}`, `{{q_max}}`, `{{ai_max}}`, `{{weight_max}}`, etc.
- Prefer `.format(**dict)`-style formatting (not f-strings) to match the exact requirement.

### Numeric literal
- **All numeric values** you place in `default_scale`, `small_scales`, and `large_scales` **must be Python/JSON-parseable numeric literals**:
  - Allowed: integers like `1000000`; floats like `0.001`; scientific notation like `1e6` or `2.5e-3`.
- Ensure **every value is a number type** (int or float) — **not** a string.
- within INT32 range (±2e9) if possible; otherwise, INT64 (±9e18).
- When these numbers appear in the rendered `template`, they should appear exactly as inserted (e.g., `1e6` or `1000000`), with no extra formatting.


### Output format (must follow exactly)
- Return **one** Python code block with:
  - Module docstring explaining what was interpreted as “data scale”.
  - The three required top-level variables: `template`, `default_scale`, `small_scales`, `large_scales`.
  - Brief inline comments justifying the chosen small/large tiers by complexity bands (e.g., “tier 1 supports O(n³) brute force because n≤300”, etc.).

### Edge cases
- If the prompt has **multiple inputs** (e.g., arrays per test case), introduce placeholders for each relevant bound (`{{n_max}}`, `{{q_max}}`, `{{ai_max}}`, …).
- If the original mixes absolute limits and typical values, set defaults to **exactly** the original absolute limits.

### Final reminder
- Your code must be immediately usable as:
```python
text = template.format(**default_scale)
# and similarly for any dict in small_scales / large_scales
```
- Lists small_scales and large_scales must each have length ≤ 3.

Every dict in those lists must be complete (all placeholders provided).

Now read the input section of the problem statement below and produce the single Python file as specified.

**Input Section:**
{problem}

'''


extract_generator_prompt = '''
You are given as input a single C++ source file that implements a *problem-generator* for an algorithmic contest. Your job is to **emit one Python code block** that parameterizes key numeric constants in that C++ code and allows reconstructing the original code via `.format(**default_scale)`.

## Input
- A raw C++ generator program (as plain text).
- A raw Python dictionary `default_scale` that maps placeholder names to their original numeric values in the C++ code.

## Output (emit exactly one Python code block):
Produce Python code that defines **only** the following objects (and optional helper comments).

`generator_code: str`
   - A **Python format template string** (not a function) that contains the entire original C++ source.
   - Replace only the relevant **hard-coded numeric constants** that are *problem-size caps / default bounds* with named placeholders using Python `.format` syntax, e.g. `{{n_max}}`, `{{m_cap}}`.
   - Keep **all other text, spacing, and newlines identical** to the original C++ (so that formatting reproduces it byte-for-byte).
   - If the C++ contains multiple occurrences of the same cap intended to be the *same semantic cap*, use the **same placeholder name** for all those occurrences.
   - Replace only when the constant is correspond to one of the items in `default_scale`.
   
## Strict requirements
generator_code must be a string template, not a function. It must be usable as:

```python
  restored = generator_code.format(**default_scale)
```
and restored must be byte-identical to the original input C++.

Now, read the provided C++ generator program, the python dictionary `default_scale` and output the single Python code block accordingly.

**C++ generator program:**
```cpp
{case_code}
```

**default_scale dictionary:**
```python
{default_scale}
```
'''


extract_validator_prompt = '''
You are given as input a single C++ source file that implements a *validator / input checker* for an algorithmic contest problem (e.g., based on testlib or a custom parser). Your job is to **emit one Python code block** that parameterizes key numeric constraints in that C++ code and allows reconstructing the original code via `.format(**default_scale)`.

## Input
- A raw C++ validator program (as plain text).
- A raw Python dictionary `default_scale` that maps placeholder names to the original numeric values in the C++ code.

## Output (emit exactly one Python code block):
Produce Python code that defines **only** the following objects (and optional helper comments).

`validator_code: str`
  - A **Python format template string** (not a function) that contains the entire original C++ source **verbatim**.
  - Replace only the relevant **hard-coded numeric constants** that define input constraints / bounds / caps with named placeholders using Python `.format` syntax, e.g. `{{n_min}}`, `{{n_max}}`, `{{m_cap}}`, etc.
  - Typical constants to parameterize include: ranges in `readInt` / `readLong` / `readString` (min/max), array sizes, graph size caps (`n`, `m`), degree limits, value bounds for elements, string length limits, coordinate bounds, alphabet sizes, and any duplicated occurrences of the same semantic limit (including in error messages) that should stay synchronized.
  - Keep **all other text, spacing, and newlines identical** to the original C++ (so that formatting reproduces it byte-for-byte).
  - If the C++ contains multiple occurrences of the same *semantic constraint*, use the **same placeholder name** for all those occurrences.

## Strict requirements
validator_code must be a string template, not a function. It must be usable as:

```python
  restored = validator_code.format(**default_scale)
```
and restored must be byte-identical to the original input C++.

Now, read the provided C++ validator program, the python dictionary `default_scale` and output the single Python code block accordingly.

**C++ validator program:**
```cpp
{case_code}
```

**default_scale dictionary:**
```python
{default_scale}
```
'''

generator_cmd_prompt= '''
You are given the following input:

1. A C++ code, which is compiled into an executable file `./gen`. This code generates test cases for a competitive programming problem.
2. A JSON object configuration, which outlines the problem's data size constraints. This configuration may reflect limits and conditions that are mentioned in the problem description.
3. The data size constraints are crucial because they control the time complexity necessary to solve the problem, thus influencing the difficulty level. I will provide you with a list of JSON configurations, each corresponding to a different data size, arranged from smallest to largest.

Your task is to generate a series of commands. These commands will execute the `./gen` file and will be wrapped in ```bash``` tags. Specifically, you must organize the commands into groups:

- Each group should contain **20 commands** with different CLI arguments, and each group corresponds to a configuration in the list, sorted in ascending order of data size.
- For example:
    - When testing with the smallest configuration, the first **20 commands** should be generated.
    - For the second configuration (slightly larger), generate **40 commands** (first 40 commands in sequence).
    - For the third configuration, generate **60 commands**, and so on, ensuring that the groups grow in size according to the increasing configurations.
- Insert **one single-line comment** between consecutive groups as a separator (e.g., `# ----- Group k: <brief note> -----`).
- For every command, ensure it differs from others in its **CLI arguments** so we get diverse test cases within and across groups.

**Key Requirements:**
1. **Data Coverage for Edge Cases**: Regardless of which data size I select, ensure that the commands **cover all possible edge cases** for the problem. The commands should stress test the problem's limits and check the **maximum values** for each configuration.
2. The data size of CLI arguments should be arranged from smallest to largest. 
3. The commands should allow me to test across different levels of problem difficulty, ensuring that each configuration produces the necessary data to assess the problem's complexity and behavior across a range of sizes.
4. **Group Separation**: Each group of commands should be separated by a **single-line comment** to clearly distinguish between different groups.

**Output Format:**
```bash
# ----- Group 1: <note> -----
./gen <args_for_cmd_1>
...
./gen <args_for_cmd_20>
# ----- Group 2: <note> -----
./gen <args_for_cmd_21>
...
./gen <args_for_cmd_60>
...
```

Now, read the provided C++ generator program, the JSON object configuration,the list of JSON configurations of different data sizes and output the corresponding bash commands in ```bash ```.

** C++ generator program:**
```cpp
{case_code}
```
** JSON object configuration:**
```json
{default_scale}
```
** List of JSON configurations of different data sizes:**
```json
{scale_list}
```
'''

generator_40cmd_prompt = """
You are given a *single* C++ source file that uses **testlib** and has already been compiled to an executable `./gen`. That program generates test cases for a competitive programming problem.

Your job: **emit exactly forty (40) shell commands** that each run `./gen` with thoughtfully chosen CLI arguments so that the resulting 40 test files together provide broad, rigorous coverage:
- edge cases (minimums, zeros, degenerate structures),
- typical cases,
- adversarial and pathological cases,
- **stress tests hitting configured maxima**,
- and diverse distributions/structures implied by the generator.

### What you must read & infer from the code
1. **Parse CLI options** by scanning for `opt<T>("name", ...)`, default values, enums/strings (e.g., `type`), and any derived constraints (caps, clamping, assertions).
2. **Respect hard limits** found in code (e.g., `n = min(n, 2e5)`, `m = min(m, n*(n-1)/2)`, `assert(1 <= n && n <= 2e5)`, etc.). Never exceed them.
3. Note any **coupled constraints** (e.g., `m` depends on `n`, connectivity flags, component counts, value ranges, parity constraints, sortedness flags).

### Coverage plan (use as a checklist)
Design the 40 commands to collectively cover the following categories (tailor names to the actual flags in the code):
- **Min/degenerate**: smallest valid `n`, `m`, counts = 0/1, empty/singleton structures.
- **Small structured**: lines/paths, stars, cliques, grids, simple cycles, trees, bipartite, etc. (if supported by `type` or toggles).
- **Medium random**: moderate sizes with uniform or skewed distributions; different `type`/mode values.
- **Corner-value ranges**: min/max for weights/values/coordinates, negative/zero/positive if allowed; duplicates/ties; sortedness on/off.
- **Adversarial**: worst-case shapes for common algorithms (e.g., long chains, high diameter, many components, near-dense graphs, heavy duplicate keys), respecting constraints.
- **Parameter cross-product**: combine flags that interact (e.g., `connected=0/1`, `distinct=0/1`, `directed=0/1`, `weighted=0/1`, etc.).
- **Boundary-near**: just below/at maxima (e.g., `n = n_max-1`, `n = n_max`; `m` near caps).
- **Stress maxima**: hit maximum feasible sizes allowed by the code and constraints.
- **Invalid-guarded fronts**: if the generator internally clamps or rejects, choose values that *exercise* those branches while still producing valid output.

### Output requirements (strict)
- **Output exactly 40 commands**.
- **Wrap them in a single fenced code block labeled `bash`**.
- Each line must be a command that starts with `./gen` followed only by **valid CLI flags that the code supports** (e.g., `-n`, `-m`, `-type`, `-seed`, and any others discovered).
- **No extra commentary, no blank lines, no redirections**, no `mkdir`, no `echo`, no shebangs—**just the 40 `./gen ...` lines**.
- Do **not** invent flags; only use those present in the code.
- **Respect all assertions and clamping logic**; never request an infeasible parameter combination.

### IMPORTANT
Return **only** one fenced code block:
```bash
./gen -flag1 ...
...
(40 lines total)

** C++ generator program:**
```cpp
{case_code}
```
"""


generate_logic_problem_prompt = """
You are given a problem statement from an algorithmic competition. 
Your task is to write a Python function `generate_logic_problem(test_case)` that:

- Takes a single test case as input, where `test_case` is a string representing the actual input for the problem.
- Parses the `test_case` string to extract the necessary values and context from the original problem statement.
- Returns a **logic problem** that focuses on the core of the original problem statement, including **key requirements, constraints**, and the **question** that needs to be answered.
- The returned string should be phrased as a **logical reasoning question**, clearly referencing the test case and incorporating the original input values.

### Requirements:
1. **Input Format**:
   - The function will receive one valid test case as a string.
   - The test case is represented as a string containing the input in the format specified by the competition. If there are multiple test cases, only one test case will be given in the input string.

2. **Output Format**:
   - The function must return a **logical reasoning question** based on the parsed values and problem context.
   - The question should include **key details** necessary for solving the problem.
   - The question should focus on the **core logic of the problem** and **must directly lead to the same correct solution** as the original problem’s solution would if the problem were solved using standard methods.

3. **Assumptions**:
   - The input test case is always valid and provided as a string.
   - The function handles only **one test case per call**.
   - The question generated must ensure that when the test case is solved using the correct approach, the answer matches the expected output.

4. **Important Notes**:
   - You **do not need to include code-specific details**, such as example code, or instructions about how the algorithm is implemented.
   - **Focus only on what is logically needed** to answer the question and ensure that the question leads to the **same answer** that would be found by solving the original problem using correct methods.
   - The question must be **coherent** and **relevant** to the provided input and **remain logically consistent** with the problem’s constraints.

### Example Problem Formats:

#### Example 1: Simple Sum
- **Problem Statement**: Given an array of integers, return the sum.
- **Test Case**: `"3\n1 2 3"`
- **Logic Question**: "What is the sum of the integers in the array [1, 2, 3]?"
- **Expected Answer**: 6

#### Example 2: Multiple Test Cases
- **Problem Statement**: Given an array of integers, return the sum for each test case.
- **Test Case**: `"1\n3\n1 2 3"`
- **Logic Question**: "What is the sum of the integers in the array [1, 2, 3]?"
- **Expected Answer**: 6

### Task:
Write a Python function `generate_logic_problem(test_case: str) -> str` that follows the above guidelines:
- Parse the `test_case` string to extract the input values.
- Use the **necessary context** from the problem statement, including **key requirements and constraints**, to generate a clear and logical question.
- Ensure that the logic question **leads to the same correct solution** as solving the original problem.
- If parsing is successful, return a **relevant and coherent** logic question based on the extracted values.
- If an error occurs during parsing, return None.
- Ensure the question is directly tied to the **logical reasoning** needed to solve the problem and guarantees the **correct answer**.

**Problem Statement:**
{problem}
"""



generate_test_case_prompt = '''
You are given a problem statement from an algorithmic competition. Your task is to generate a list of valid test cases for this problem, in the format of a JSON array. Each element in the array should be an object with a key 'test_case' and the corresponding value should be a string representing a valid input for the problem. The input should follow the problem description, and for problems that involve multiple test cases, each input should be a single test case represented as a string. 

### Specific requirements:
1. Each test case should contain a single, valid input that adheres to the problem statement.
2. If the problem involves multiple test cases, the input should include the count of test cases (1) as the first line.
3. The number of tokens in each test case should not exceed 200 tokens.
4. The test cases in the list should be ordered from simplest to most complex.
  
### Example format for the output (JSON array):
```json
[
  {{
    "test_case": "input1"
  }},
  {{
    "test_case": "input2"
  }},
  ...
]
```
The input problem statement may include various types of data, such as integers, strings, arrays, etc., and you should ensure the test cases conform to these formats.

**Problem Statement:**
{problem}
'''

generate_generator_prompt = """
Please write a Python function named `generator()` that generates random, valid test case inputs for an algorithmic problem. The function should generate valid test cases, ensuring that the generated input strictly adheres to the problem's constraints and requirements.

The generator function should follow these guidelines:

1. **Test Case Structure**: 
   - For problems with multiple test cases, the number of test cases must be 1.
   - Each test case must be valid and satisfy the exact constraints specified in the problem.

2. **Randomized Data Types**: 
   - Ensure the generator selects appropriate random data types based on the problem.
     - For graph problems, the generator should randomly select different graph structures, such as trees, dense graphs, sparse graphs, etc.
     - For mathematical problems, the generator should create valid inputs, including edge cases (e.g., out-of-range values or -1 for error cases).
   
3. **Constraints on Values**: 
   - Ensure that all numeric values generated are less than {max_number}. No number should exceed this threshold.

4. **Output Format**: 
   - The function should return the generated test case as a string, formatted correctly.

**Function Signature:**
```python
def generator():
    # Your code here
```
**Problem Statement:**
{problem}
"""

scale_param_extractor_prompt ='''
You are given as input the full statement of a single algorithmic problem. Your job is to **emit one JSON code block** that extracts all numeric *scale parameters* that are relevant to the **time complexity** of typical solutions.

A *scale parameter* is any integer quantity that bounds:
- The number of items, elements, or positions (e.g. `n` = number of elements, `m` = number of edges, `q` = number of queries).
- The size of a grid, string, or sequence (e.g. length up to `2e5`, grid up to `1000 × 1000`).
- The size of state space or iteration space that an algorithm must explicitly handle.

## Input
- A raw problem statement in natural language. It may include:
  - Formal constraints section (e.g. “1 ≤ n ≤ 2⋅10^5”).
  - Definitions of variables (e.g. “The first line contains an integer n — the number of vertices.”).
  - Multiple types of constraints interleaved in the text.

## Output (emit exactly one JSON code block)
Produce **only one** JSON object in a fenced JSON code block. The JSON must map parameter names to an object of the form:

```json
{{
  "n": {{ "max": 100000, "min": 2 }},
  "m": {{ "max": 200000, "min": 0 }}
}}
```
## What to INCLUDE
Include a parameter only if all of the following are true:
- It directly bounds the size or count of something that is iterated over, e.g.: 
  - number of elements / vertices / edges / queries (n, m, q, etc.),
  - length of a string or array, 
  - rows / columns of a grid.
- The bounds appear explicitly in the statement, usually in the Input / Constraints section.

## What to EXCLUDE
- Number of test cases / groups: never include t, T, or similar when it means “number of test cases”.
- Pure value ranges for single items that do not change the input size, e.g.:
  - -10^9 ≤ a_i ≤ 10^9 when a_i is just the value of an element.
  - Coordinate or weight ranges that are not used as sizes of arrays/grids.
- Any quantity that only affects output format or precision.

Now, read the provided problem statement and output the single JSON code block accordingly.
{problem}
'''

testcase_generator_prompt = '''
You are given as input a single *algorithmic problem statement* (like those from programming contests). Your job is to **emit one Python code block** that defines a *test-case generator* function for this problem.

The generator must produce **exactly one** valid test case per call, parameterized only by the numeric scale values provided via a JSON object.

## Input
You will be given:

1. A raw problem statement in natural language that fully specifies:
   - the input format,
   - the constraints,
   - and the meaning of each variable.

2. An example `json_obj` instance:
   - This is only an example to clarify field names and typical ranges.
   - Your code must work for any valid `json_obj` that matches the described schema.

## Required Python output (emit exactly one Python code block)
You must output a Python code block that defines **one single function** with the following signature:

```python
def generate_testcase(json_obj: dict) -> tuple[str, dict]:
    """
    Generate a test case based on the given json_obj.

    Parameters:
    - json_obj (dict): The input JSON object containing problem parameters.

    Returns:
    - tuple[str, dict]: A tuple containing:
      - The first element is a string representing the test case in input format.
      - The second element is a dictionary representing the same test case.
    """
    ...
```

### Return value
- Your function must return both the string and the dictionary representation of the test case in a tuple. The first element of the tuple should be the string format, and the second element should be the dictionary format.
  - output_str:
    - A single string that is a valid input for the problem according to the Input section, representing exactly one logical test case.
    - If the problem statement defines a format with multiple test cases controlled by an integer T in the input,You must set T = 1.
    - Example:"1\n5\n1 2 3 4 5"
  - output_dict :
    - A Python dict that is a structured, formal description of the same test case. 
    - If the problem statement contains multiple test cases, **do not** introduce T or any extra wrapper.
    - Example:{{"n":5,"list":[1,2,3,4,5]}}

## Constraints
- All sizes (counts, lengths, number of operations, etc.) must be determined only from json_obj.
- All other values (elements of arrays, weights, edges, indices, etc.) must be generated randomly within a reasonable range and **strictly smaller than 10000**, while satisfying the problem’s constraints at the same time.
- If the problem allows "no-solution" cases (e.g., the intended output is -1 when no solution exists), you should **bias your random generation towards test cases that admit at least one valid solution**, and explicitly construct values to satisfy any hidden feasibility conditions, so that the correct solution is not trivially always the "no-solution" output.

## Problem statement
{problem}
## Example json_obj
{example_json_obj}
'''
answer_problem_prompt = '''
{problem}
Please reason step by step, and put your final answer within \\boxed{{}}.
'''

def train_prompt(question,choices=None):
  system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, and put your final answer within \\boxed{{}} . The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>." 
  if choices: question = question + "\n" + choices
  return [
    {"content":system_prompt, "role": "system"},
    {"content": question, "role": "user"}
  ]

def no_think_prompt(question,choices=None):
  system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." 
  if choices: question = question + "\n" + choices
  return [
    {"content":system_prompt, "role": "system"},
    {"content": question, "role": "user"}
  ]



problem_meta_extractor_prompt = '''
You are given as input the full statement of a single algorithmic problem. Your job is to **emit one JSON code block** that captures:

1. All numeric *scale parameters* that are relevant to the **time complexity** of typical solutions.
2. The **type** of the required output.
3. Whether the required output is **unique**, i.e., whether the problem has exactly one correct output for each valid input.

## 1. Scale parameters

A *scale parameter* is any integer quantity that bounds:
- The number of items, elements, or positions (e.g. `n` = number of elements, `m` = number of edges, `q` = number of queries).
- The size (length) of a grid, string, or sequence (e.g. length up to `2e5`, grid up to `1000 × 1000`).
- The size of a state space or iteration space that an algorithm must explicitly handle.

### What to INCLUDE as scale parameters
Include a parameter only if **all** of the following are true:
- It directly bounds the size or count of something that is iterated over, e.g.:
  - number of elements / vertices / edges / queries (`n`, `m`, `q`, etc.),
  - length of a string or array,
  - rows / columns of a grid.
- The bounds appear explicitly in the statement, usually in the Input / Constraints section, like:
  - `1 ≤ n ≤ 2⋅10^5`
  - `0 ≤ m ≤ 2⋅10^5`
  - `1 ≤ |s| ≤ 1000`
  - `1 ≤ n, m ≤ 500`
- You can clearly identify the parameter name (e.g., `n`, `m`, `q`, `k`, `N`, etc.).

### What to EXCLUDE from scale parameters
- Number of test cases / groups: **never** include `t`, `T`, or similar when it means “number of test cases”.
- Pure value ranges for single items that do **not** change the input size, e.g.:
  - `-10^9 ≤ a_i ≤ 10^9` when `a_i` is just the value of an element.
  - Coordinate or weight ranges that are not used as sizes of arrays/grids.
- Any quantity that only affects output format or precision.

### Representation of scale parameters
In the JSON, represent scale parameters under the key `"scale_params"` as:

```json
"scale_params": {{
  "n": {{ "min": 1, "max": 200000 }},
  "m": {{ "min": 0, "max": 200000 }}
}}
```

## 2. Output type classification
You must classify the type of the required output into exactly one of the following strings:
- "string": The required output is a single string or a small number of strings.
- "number": The required output is a single numeric value (integer, real, etc.), e.g. “print one integer — the answer”.
- "array": The required output is a one-dimensional sequence (list) of values, e.g. an array of integers, a permutation, a sequence of answers for each query when printed as space-separated numbers or in multiple lines.
- "graph": The required output is a graph structure, such as a set of edges, tree description, adjacency list, or any structure where the output itself is naturally a graph.
- "matrix": The required output is a 2D grid or matrix (e.g. n × m numbers, characters, or cells).
- "bool": The required output is logically a boolean answer, e.g. “YES/NO”, “True/False”, “possible/impossible”, "Alice"/"Bob", etc.
- "others": The required output is a complex or mixed structure (e.g. several heterogeneous values, or text explanations) that does not fit clearly into any of the above categories.

## 3. Output uniqueness
You must also decide whether the required output is unique for each valid input.
Define "is_output_unique" as:
- true if, for any fixed valid input, there is exactly one correct output that satisfies the problem statement.
- false if the statement allows multiple different outputs to be accepted as correct for the same input.

## JSON Output Specification
You must produce exactly one JSON object in a fenced JSON code block.
The JSON must have the following top-level keys:
- "scale_params": an object mapping parameter names to {{ "min": <int>, "max": <int> }}.
- "output_type": one of "string", "number", "array", "graph", "matrix", "bool", "others".
- "is_output_unique": a boolean.

## Example 1 (with scale parameters and yes/no output)
```json
{{
  "scale_params": {{
    "n": {{ "min": 1, "max": 200000 }},
    "m": {{ "min": 0, "max": 200000 }}
  }},
  "output_type": "bool",
  "is_output_unique": True
  
}}
```

## Example 2 (no scale parameters, numeric non-simple output)
```json
{{
  "scale_params": {{}},
  "output_type": "number",
  "is_output_unique": true
}}
```

## Example 3 (multiple valid outputs allowed)
```json
{{
  "scale_params": {{
    "n": {{ "min": 1, "max": 100000 }}
  }},
  "output_type": "array",
  "is_output_unique": false
}}
```

## Final instruction
Now, read the provided problem statement and output the single JSON code block accordingly.

{problem}
'''
if __name__ == "__main__":
    # example usage
    case_code = "print(42)"
    problem = "what your name?"
    print(upgrade_prompt.format(code=case_code))
    print(extract_number_prompt.format(problem=problem))

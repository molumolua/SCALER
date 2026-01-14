
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


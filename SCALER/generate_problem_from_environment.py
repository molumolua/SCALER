try:
    from after_extract import generate_problem_detail_and_ground_truth,str_to_float_list_or_none
    from logger import setup_logger
except:
    from SCALER.after_extract import generate_problem_detail_and_ground_truth,str_to_float_list_or_none
    from SCALER.logger import setup_logger
def get_problems(example, input_difficultys, sandboxfusion_url, logger=None,
                 max_try=3, with_instruction=True):
    template = '''
# {problem_name} Problem Description:
{description}

# Input Instance: 
{problem_detail}
'''
    contents = []
    output_type = example.get('output_type',"number")  # "string" / "array" / None / 其他

    for input_difficulty in input_difficultys:
        problem_scale = example['difficulty_dict'][str(input_difficulty)]
        problem_scale_dict = {
            k: int(problem_scale * v.get('base',1.0) + v['min'])
            for k, v in example['params'].items()
            if k != 'difficulty'
        }

        problem_detail, ground_truth = generate_problem_detail_and_ground_truth(
            example, problem_scale_dict, sandboxfusion_url, logger=logger
        )

        content = template.format(
            problem_name=example['name'],
            description=example['logic_description'],
            problem_detail=problem_detail
        )

        if with_instruction:
            example_instruction = example.get("instruction", "")
            meta_instruction = '''
Treat the description as a single fully specified math/algorithm problem about the given concrete input, and ignore any mentions of writing a program, input/output formats, or multiple test cases. Reason step by step to compute the required answer for this instance.
'''.strip()


            format_instruction = ""
            if output_type == "array":
                format_instruction = ("Output all the required numerical answers in \\boxed{{[]}} in the form of one-dimensional LaTeX array, e.g. \\boxed{{[\\pi, \\frac{{1}}{{2}}, 3]}}.")
                # format_instruction = ( "Output all the required numerical answer in the \\boxed{{}} in the form of one-dimensional JSON array ." )
                # format_instruction = (
                #     "Output all the required output in the form of one-dimensional JSON array, e.g. [1,2,3] in \\boxed{{}}."
                # )
                ground_truth = str_to_float_list_or_none(ground_truth)
            
            if output_type =='string':
                ground_truth = ground_truth.split()[0]
                

            content += f"""
# Instruction
{example_instruction}
{meta_instruction}
{format_instruction}
"""
        

        contents.append((content, f"\\boxed{{{ground_truth}}}"))

    return contents

if __name__ == "__main__":
    logger=setup_logger()
    example={
    "199_C. About Bacteria": {
        "name": "199_C. About Bacteria",
        "logic_description": "Qwerty the Ranger took up a government job and arrived on planet Mars. He should stay in the secret lab and conduct some experiments on bacteria that have funny and abnormal properties. The job isn't difficult, but the salary is high.\n\nAt the beginning of the first experiment there is a single bacterium in the test tube. Every second each bacterium in the test tube divides itself into k bacteria. After that some abnormal effects create b more bacteria in the test tube. Thus, if at the beginning of some second the test tube had x bacteria, then at the end of the second it will have kx + b bacteria.\n\nThe experiment showed that after n seconds there were exactly z bacteria and the experiment ended at this point.\n\nFor the second experiment Qwerty is going to sterilize the test tube and put there t bacteria. He hasn't started the experiment yet but he already wonders, how many seconds he will need to grow at least z bacteria. The ranger thinks that the bacteria will divide by the same rule as in the first experiment. \n\nHelp Qwerty and find the minimum number of seconds needed to get a tube with at least z bacteria in the second experiment.",
        "raw_description": "Qwerty the Ranger took up a government job and arrived on planet Mars. He should stay in the secret lab and conduct some experiments on bacteria that have funny and abnormal properties. The job isn't difficult, but the salary is high.\n\nAt the beginning of the first experiment there is a single bacterium in the test tube. Every second each bacterium in the test tube divides itself into k bacteria. After that some abnormal effects create b more bacteria in the test tube. Thus, if at the beginning of some second the test tube had x bacteria, then at the end of the second it will have kx + b bacteria.\n\nThe experiment showed that after n seconds there were exactly z bacteria and the experiment ended at this point.\n\nFor the second experiment Qwerty is going to sterilize the test tube and put there t bacteria. He hasn't started the experiment yet but he already wonders, how many seconds he will need to grow at least z bacteria. The ranger thinks that the bacteria will divide by the same rule as in the first experiment. \n\nHelp Qwerty and find the minimum number of seconds needed to get a tube with at least z bacteria in the second experiment.\n\nInput\n\nThe first line contains four space-separated integers k, b, n and t (1 ≤ k, b, n, t ≤ 106) — the parameters of bacterial growth, the time Qwerty needed to grow z bacteria in the first experiment and the initial number of bacteria in the second experiment, correspondingly.\n\nOutput\n\nPrint a single number — the minimum number of seconds Qwerty needs to grow at least z bacteria in the tube.\n\nExamples\n\nInput\n\n3 1 3 5\n\n\nOutput\n\n2\n\nInput\n\n1 4 4 7\n\n\nOutput\n\n3\n\nInput\n\n2 2 4 100\n\n\nOutput\n\n0",
        "solutions": {
        "solution": [
            "#include <bits/stdc++.h>\nusing namespace std;\nusing namespace std;\nint main() {\n  long long int k, b, n, t;\n  while (cin >> k >> b >> n >> t) {\n    long long int s = 1;\n    long long int cas = 0;\n    while (s <= t && cas < n) {\n      s = s * k + b;\n      cas++;\n    }\n    if (cas == n && s <= t)\n      cout << 0 << endl;\n    else\n      cout << n - cas + 1 << endl;\n  }\n  return 0;\n}\n",
            "#------------------------template--------------------------#\nimport os\nimport sys\nfrom math import *\nfrom collections import *\nfrom fractions import *\nfrom bisect import *\nfrom heapq import*\nfrom io import BytesIO, IOBase\ndef vsInput():\n    sys.stdin = open('input.txt', 'r')\n    sys.stdout = open('output.txt', 'w')\nBUFSIZE = 8192\nclass FastIO(IOBase):\n    newlines = 0\n    def __init__(self, file):\n        self._fd = file.fileno()\n        self.buffer = BytesIO()\n        self.writable = \"x\" in file.mode or \"r\" not in file.mode\n        self.write = self.buffer.write if self.writable else None\n    def read(self):\n        while True:\n            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))\n            if not b:\n                break\n            ptr = self.buffer.tell()\n            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)\n        self.newlines = 0\n        return self.buffer.read()\n    def readline(self):\n        while self.newlines == 0:\n            b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))\n            self.newlines = b.count(b\"\\n\") + (not b)\n            ptr = self.buffer.tell()\n            self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)\n        self.newlines -= 1\n        return self.buffer.readline()\n    def flush(self):\n        if self.writable:\n            os.write(self._fd, self.buffer.getvalue())\n            self.buffer.truncate(0), self.buffer.seek(0)\nclass IOWrapper(IOBase):\n    def __init__(self, file):\n        self.buffer = FastIO(file)\n        self.flush = self.buffer.flush\n        self.writable = self.buffer.writable\n        self.write = lambda s: self.buffer.write(s.encode(\"ascii\"))\n        self.read = lambda: self.buffer.read().decode(\"ascii\")\n        self.readline = lambda: self.buffer.readline().decode(\"ascii\")\nsys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)\ninput = lambda: sys.stdin.readline().rstrip(\"\\r\\n\")\nALPHA='abcdefghijklmnopqrstuvwxyz'\nM=10**9+7\ndef value():return tuple(map(int,input().split()))\ndef array():return [int(i) for i in input().split()]\ndef Int():return int(input())\ndef Str():return input()\ndef arrayS():return [i for i in input().split()]\n\n\n#-------------------------code---------------------------#\n# vsInput()\n\ndef canMake(iter):\n    pass\n\n\nk,b,n,t=value()\nz=1\n\nreduced=0\n\n# for i in range(n):\n#     z=z*k+b\n#     print(z)\n# print()\n# z=t\n# for i in range(n):\n#     z=z*k+b\n#     print(z)\n\nwhile(z<=t):\n     reduced+=1\n     z=z*k+b\n\nprint(max(0,n-reduced+1))\n\n    \n\n\n\n\n    \n\n\n\n\n\n\n\n\n\n\n\n\n\n                \n\n    \n\n\n\n\n\n\n\n\n\n    \n",
            "#include <bits/stdc++.h>\nusing namespace std;\nlong long int dir[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};\nvoid solve() {\n  long long int k, b, n, t;\n  cin >> k >> b >> n >> t;\n  long long int ans = 0, bacteria = k + b;\n  while (bacteria <= t) {\n    bacteria = (bacteria * k) + b;\n    ans++;\n  }\n  cout << max(0LL, n - ans);\n}\nint main() {\n  ios_base::sync_with_stdio(false);\n  cout.tie(NULL);\n  cin.tie(NULL);\n  ;\n  long long int t = 1;\n  while (t--) {\n    solve();\n  }\n  return 0;\n}\n",
            "#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n  long long k, b, t, n, z, ans;\n  cin >> k >> b >> t >> n;\n  z = 1;\n  int i = 0;\n  while (z <= n) {\n    z = k * z + b;\n    i++;\n  }\n  ans = t - i + 1;\n  if (ans < 0) ans = 0;\n  cout << ans << endl;\n  return 0;\n}\n",
            "#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n  long long k, b, n, t;\n  cin >> k >> b >> n >> t;\n  long long now = 1, ans = 0;\n  for (long long i = 1; i <= n; i++) {\n    if (now * k + b > t) break;\n    now = now * k + b, ans++;\n  }\n  cout << n - ans;\n}\n",
            "#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n  ios::sync_with_stdio(0);\n  cin.tie(NULL);\n  long long k, b, n, t;\n  cin >> k >> b >> n >> t;\n  int x = 1;\n  while (k * x + b <= t) x *= k, x += b, n--;\n  cout << max((long long)0, n);\n}\n",
            "#include <bits/stdc++.h>\nusing namespace std;\nconst double eps = 1e-9;\nconst double pi = acos(-1.0);\nconst int maxn = (int)1e5 + 10;\nconst int mod = (int)1e9;\nint fastMax(int x, int y) { return (((y - x) >> (32 - 1)) & (x ^ y)) ^ y; }\nint fastMin(int x, int y) { return (((y - x) >> (32 - 1)) & (x ^ y)) ^ x; }\nint main() {\n  long long k, b, n, t, z;\n  cin >> k >> b >> n >> t;\n  z = 1;\n  while (t >= z * k + b) {\n    z = z * k + b;\n    n--;\n  }\n  cout << (n > 0 ? n : 0) << endl;\n  return 0;\n}\n",
            "#include <bits/stdc++.h>\nusing namespace std;\nconst int INF = 1000000009;\nconst double PI = acos(-1.0);\nconst double eps = 1e-8;\nconst int MAXN = 0;\nconst int MAXM = 0;\nlong long n, k, t, b, z;\nint main() {\n  z = 1;\n  int ans = 0;\n  cin >> k >> b >> n >> t;\n  while (z <= t) {\n    ans++;\n    z = z * k + b;\n  }\n  if (z == t)\n    ans = n - ans;\n  else {\n    ans--;\n    ans = n - ans;\n  }\n  cout << max(0, ans) << endl;\n  return 0;\n}\n",
            "#include <bits/stdc++.h>\nusing namespace std;\nint main() {\n  long long k, b, n, t;\n  cin >> k >> b >> n >> t;\n  long long now = 1, res = n;\n  if (k == 1) {\n    long long nn = 1 + n * b - t;\n    if (nn <= 0) {\n      puts(\"0\");\n      return 0;\n    }\n    long long dd = b;\n    long long rr;\n    if (nn % dd)\n      rr = nn / dd + 1;\n    else\n      rr = (nn + dd - 1) / dd;\n    cout << max(rr, 0LL) << endl;\n    return 0;\n  }\n  long long nn = t * (k - 1) + b;\n  long dd = (k - 1) + b;\n  long long to;\n  if (nn % dd)\n    to = nn / dd;\n  else\n    to = (nn + dd - 1) / dd;\n  while (res >= 0) {\n    now *= k;\n    if (now <= to)\n      ;\n    else\n      break;\n    res--;\n  }\n  res = max(res, 0LL);\n  cout << res << endl;\n  return 0;\n}\n",
            "#include <bits/stdc++.h>\nusing namespace std;\n#pragma GCC optimize(\"O3\")\nconst long long int mod = 1e8;\nlong long int powmod(long long int x, long long int y) {\n  long long int t;\n  for (t = 1; y; y >>= 1, x = x * x % mod)\n    if (y & 1) t = t * x % mod;\n  return t;\n}\nlong long int gcd(long long int x, long long int y) {\n  return y ? gcd(y, x % y) : x;\n}\nlong long int lcm(long long int x, long long int y) {\n  return x * (y / gcd(x, y));\n}\nlong long int modd(long long int a) { return (a % mod + mod) % mod; }\ndouble findMod(double a, double b) {\n  double mods;\n  if (a < 0)\n    mods = -a;\n  else\n    mods = a;\n  if (b < 0) b = -b;\n  while (mods >= b) mods = mods - b;\n  if (a < 0) return -mods;\n  return mods;\n}\nlong long int add(long long int a, long long int b) {\n  return modd(modd(a) + modd(b));\n}\nlong long int mul(long long int a, long long int b) {\n  return modd(modd(a) * modd(b));\n}\nint smask(int i, int pos) { return (i | (1 << pos)); }\nint clmask(int i, int pos) { return (i & (~(1 << pos))); }\nbool chmask(int i, int pos) { return (i & (1 << pos)) != 0; }\ndouble cordist(pair<double, double> a, pair<double, double> b) {\n  return sqrt(((a.first - b.first) * (a.first - b.first)) +\n              ((a.second - b.second) * (a.second - b.second)));\n}\nlong long binpow(long long a, long long b) {\n  if (b == 0) return 1;\n  long long res = binpow(a, b / 2);\n  if (b % 2)\n    return mul(mul(res, res), a);\n  else\n    return mul(res, res);\n}\nlong long int n, m, q, r, i, j;\nvector<long long int> arr;\nvoid solve() {\n  long long int k, b, t;\n  cin >> k >> b >> n >> t;\n  long long int prev = 1;\n  long long int sec = 0;\n  while (prev <= t) {\n    prev = (k * prev) + b;\n    sec++;\n  }\n  cout << max(0LL, n - sec + 1) << \"\\n\";\n}\nint main() {\n  ios_base::sync_with_stdio(0);\n  cin.tie(0);\n  cout.tie(0);\n  int k = 1;\n  while (k--) {\n    solve();\n  }\n  return 0;\n}\n"
        ],
        "language": [
            2,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2
        ]
        },
        "generate_testcase": "\nimport traceback\nfrom string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\ndef generate_testcase(json_obj, output_format: str = \"str\"):\n    \"\"\"\\n    Generate a single test case for the bacterial growth problem.\\n\\n    Parameters\\n    ----------\\n    json_obj : dict\\n        Dictionary containing numeric scale values. Keys may be:\\n            - 'k', 'b', 'n', 't'          – direct upper bound for the respective variable\\n            - 'max_k', 'max_b', 'max_n', 'max_t' – alternative names for upper bounds\\n            - 'max'                      – generic upper bound used when a specific bound is missing\\n        All bounds are clamped to the problem limits (1 … 10^6).\\n    output_format : str, optional\\n        If \"str\", returns a space‑separated string suitable as program input.\\n        If \"dict\", returns a dictionary with keys 'k', 'b', 'n', 't'.\\n\\n    Returns\\n    -------\\n    str or dict\\n        The generated test case in the requested format.\\n    \"\"\"\n    import random\n\n    GLOBAL_MAX = 10 ** 6\n\n    def get_bound(var_name):\n        \"\"\"\\n        Retrieve the upper bound for a variable from json_obj.\\n        Preference order: direct name, 'max_<name>', then generic 'max'.\\n        \"\"\"\n        # Direct bound, e.g. json_obj['n']\n        bound = json_obj.get(var_name)\n        if bound is None:\n            # Alternate form, e.g. json_obj['max_n']\n            bound = json_obj.get(f\"max_{var_name}\")\n        if bound is None:\n            # Generic bound\n            bound = json_obj.get(\"max\", GLOBAL_MAX)\n        # Clamp to the problem limits\n        if bound < 1:\n            bound = 1\n        if bound > GLOBAL_MAX:\n            bound = GLOBAL_MAX\n        return bound\n\n    max_k = get_bound(\"k\")\n    max_b = get_bound(\"b\")\n    max_n = get_bound(\"n\")\n    max_t = get_bound(\"t\")\n\n    k = random.randint(1, max_k)\n    b = random.randint(1, max_b)\n    n = random.randint(1, max_n)\n    t = random.randint(1, max_t)\n\n    if output_format == \"dict\":\n        return {\"k\": k, \"b\": b, \"n\": n, \"t\": t}\n    else:\n        return f\"{k} {b} {n} {t}\"\n",
        "num_of_groups": 100,
        "max_group_siz": 1,
        "difficulty_dict": {
        "0": 1,
        "1": 3,
        "2": 4,
        "3": 5,
        "4": 8,
        "5": 11,
        "6": 18,
        "7": 28,
        "8": 44,
        "9": 70,
        "10": 111,
        "11": 177,
        "12": 282,
        "13": 451,
        "14": 722,
        "15": 1154,
        "16": 1846,
        "17": 2952,
        "18": 4723,
        "19": 7557,
        "20": 12090,
        "21": 19344,
        "22": 30950,
        "23": 49519,
        "24": 79229,
        "25": 126766,
        "26": 202825,
        "27": 324520,
        "28": 519231,
        "29": 830768
        },
        "params": {
        "n": {
            "min": 1,
            "max": 1000000
        },
        "difficulty": {
            "version": 1,
            "params": {
            "dmax": 29,
            "ema_beta": 0.0,
            "activate_function": "base"
            }
        }
        }
    }}
    print(get_problems(example["199_C. About Bacteria"],input_difficultys=[0],
                       sandboxfusion_url="https://nat-notebook-inspire.sii.edu.cn/ws-6e6ba362-e98e-45b2-9c5a-311998e93d65/project-a75d443b-88d5-4461-859f-548caa0b38a7/user-ffe43f44-3d3b-44eb-8c68-ea76d13211e5/vscode/343f415d-2080-49db-8901-0d11ad76754c/da4db590-2d65-4162-9d58-7ddf81e88f36/proxy/8080/run_code",
                       logger=logger))
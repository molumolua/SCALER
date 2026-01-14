from exec_and_verify import run_generator_with_alarm,sandboxfusion_run
from logger import setup_logger
# code = r'''
# // sum_min_max.cpp
# #include <iostream>
# #include <vector>
# #include <algorithm>
# #include <numeric>

# using namespace std;

# void solve() {
#     ios::sync_with_stdio(false);
#     cin.tie(nullptr);

#     int n;
#     if (!(cin >> n)) {                 // 没有输入时返回非零，便于排错
#         cerr << "NO_INPUT\n";
#         exit(2);
#     }
#     vector<long long> a(n);
#     for (int i = 0; i < n; ++i) cin >> a[i];

#     long long s  = accumulate(a.begin(), a.end(), 0LL);
#     long long mn = *min_element(a.begin(), a.end());
#     long long mx = *max_element(a.begin(), a.end());

#     cout << s << ' ' << mn << ' ' << mx << '\n';
# }

# int main() {
#     solve();
#     return 0;
# }


# '''

# logger = setup_logger()
# # code = code+r'''
# # if __name__ == "__main__":
# #     print(generator())
# # '''
# ret = sandboxfusion_run("https://nat-notebook-inspire.sii.edu.cn/ws-6e6ba362-e98e-45b2-9c5a-311998e93d65/project-4493c9f7-2fbf-459a-ad90-749a5a420b91/user-ffe43f44-3d3b-44eb-8c68-ea76d13211e5/vscode/5036c53a-7e0f-4cb7-8546-d1481ce410ef/0bb00492-4106-40c2-abf1-64a8b368ade8/proxy/8080/run_code", code,logger=logger,
#                         language='cpp',stdin="5\n1 2 3 4 5\n")
# print(ret)
# if ret["ok"]:
#     print("STDOUT:", ret["stdout"])
# else:
#     print("ERROR:", ret["error"])


# code = r'''
# print(None)

# '''

# logger = setup_logger()
# # code = code+r'''
# # if __name__ == "__main__":
# #     print(generator())
# # '''
# ret = sandboxfusion_run("https://nat-notebook-inspire.sii.edu.cn/ws-6e6ba362-e98e-45b2-9c5a-311998e93d65/project-4493c9f7-2fbf-459a-ad90-749a5a420b91/user-ffe43f44-3d3b-44eb-8c68-ea76d13211e5/vscode/5036c53a-7e0f-4cb7-8546-d1481ce410ef/0bb00492-4106-40c2-abf1-64a8b368ade8/proxy/8080/run_code", code,logger=logger,
#                         language='python',stdin="")
# if ret['run_result']['stdout'].startswith( 'None'):
#     print(ret)


print([1,2,3]*3)


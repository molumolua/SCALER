n, k = map(int,input().split())
l = list(map(int,input().split()))
a = 0
for i in range(len(l)):
    c = list(str(l[i]))
    s = 0
    for j in range(len(c)):
        if c[j] == "7" or c[j] == "4":
            s += 1
    if s <= k:
        a += 1
print(a)
            
    
    
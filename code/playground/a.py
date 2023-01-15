mod = 1e9 + 7


def euler(n):
    res = n
    mid = n
    i = 2
    while i * i <= mid:
        if mid % i == 0:
            res = res // i * (i - 1)
            while mid % i == 0:
                mid /= i
        i += 1
    if mid > 1:
        return int((res // mid * (mid - 1)) % mod)
    else:
        return int(res % mod)


# if __name__ == '__main__':
#     arr = list(map(int, input().split()))
#     for i in arr:
#         print(euler(i), end=" ")

if __name__ == '__main__':
    arr = []
    for i in range(400000):
        arr.append(i)
    n, t = map(int, input().split(" "))
    arr = list(map(int, input().split()))
    for i in range(t):
        cmd = input().split()
        l, r = int(cmd[1]), int(cmd[2])
        if cmd[0][0] == 'T':
            tmp = 1
            for j in range(l, r + 1):
                tmp *= arr[j - 1]
            print(euler(tmp))
        else:
            x = int(cmd[3])
            for j in range(l, r + 1):
                arr[j - 1] *= x


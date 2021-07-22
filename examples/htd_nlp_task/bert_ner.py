def find_top_k(arr, k):
    pool = set()
    for i in arr:
        pool.add(i)
    return sorted(list(pool), reverse=True)[k-1]

def qsort(arr):
    if len(arr) <= 1:
        return arr
    p = partition(arr, 0, len(arr)-1)
    qsort(arr[:p])
    qsort(arr[p+1:])
    return arr
    
def partition(arr, l, r):
    base = arr[l]
    base_idx = l
    while l<r:
        while arr[r] > base:
            r -= 1
        swap(arr, base_idx, r)
        base_idx = r
        while arr[l] < base:
            l += 1
        swap(arr, base_idx, l)
        base_idx = l
    return base_idx
            
def swap(arr, i , j):
    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
import random
import torch

class MockAlgorithm:
    def __init__(self, n, method):
        self.n = n
        self.k = 2
        self.j = 2
        self.arr = [0] * n
        if method == "reverse":
            for i in range(n):
                self.arr[i] = n - i - 1
        elif method == "sorted":
            for i in range(n):
                self.arr[i] = i
        elif method == "randint":
            torch.random.manual_seed(0)
            rand_array = torch.randint(0, n, (1, 1 << 14), device="cuda", dtype=torch.int16)
            self.arr = rand_array.tolist()[0]
        
    def step(self):
        if self.j == 0:
            self.k *= 2
            self.j = self.k // 2
        else:
            self.j //= 2
        if self.k > self.n:
            return True
        for i in range(self.n):
            l = i ^ self.j
            if l > i:
                if ((i & self.k) == 0 and self.arr[i] > self.arr[l]) or ((i & self.k) != 0 and self.arr[i] < self.arr[l]):
                    self.arr[i], self.arr[l] = self.arr[l], self.arr[i]
        return False

    def step_n(self, n):
        for _ in range(n):
            if self.step():
                return True
        return False

    def check(self):
        for i in range(self.n - 1):
            if not self.arr[i] <= self.arr[i + 1]:
                return False
        return True
    
def old(n):
    torch.random.manual_seed(0)
    rand_array = torch.randint(0, n, (1, n), device="cuda", dtype=torch.int16)
    arr = rand_array.clone().tolist()[0]
    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            for i in range(n):
                l = i ^ j
                if l > i:
                    if ((i & k) == 0 and arr[i] > arr[l]) or ((i & k) != 0 and arr[i] < arr[l]):
                        arr[i], arr[l] = arr[l], arr[i]
            j //= 2
        k *= 2

    for i in range(n - 1):
        assert arr[i] <= arr[i + 1]

m1 = MockAlgorithm(4096 * 2, "randint")

if __name__ == "__main__":
    while not m1.step():
        print(m1.j, m1.k)
        print(m1.arr[:8])
        pass

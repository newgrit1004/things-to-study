## Table of Contents

- [분할 상환 분석(Amortized Analysis)](#1)
- [프리발즈 알고리즘(Freivalds' algorithm)](#2)
- [밀러-라빈 소수 판별법](#2)
- [Quick Selection](#3)


## #1
### 분할 상환 분석(Amortized Analysis)
컴퓨터 공학에서 분할 상환 분석(amortized analysis)란 주어진 알고리즘의 시간 복잡도나 프로그램을 수행하는데 소요되는 시간 또는 메모리 같은 자원 사용량을 분석하기 위해서 사용하는 기법이다. amortized time의 의미는 특정 연산을 많이 반복하였을 때, 평균적으로 연산당 소요되는 시간을 의미한다. 특정 연산을 백만번 시도하였을 때 고려되는 요소는 해당 연산의 worst-case나 best-case가 아니라, 전체 소요된 시간을 신경쓰게 될 것이다. 따라서 해당 연산이 아주 가끔 느린 현상이 발생하는 것은 크게 신경쓰지 않는다. 예를 들어, dynamic array(동적 배열)에 반복적으로 새로운 item을 넣는 경우를 가정해본다. 일반적으로, 새로운 item을 추가하는 작업은 O(1)의 시간복잡도를 가진다. 하지만, array가 가득 찰 때마다 2배의 공간을 할당하고, 데이터를 새 영역에 복사하고, 이전 메모리를 free해주는 작업을 해야하며, 이러한 작업을 더블링(Doubling)이라 한다. 새로운 메모리 공간을 할당하고, 이전 메모리를 free해주는 작업은 현재 길이가 n인 array에 대해서 O(n)이 걸린다. Amortized Analysis를 통해 동적 배열에 데이터 추가하는 작업의 비용을 계산한다면 더블링이 발생하는 최악의 경우를 여러 번에 나눠서 시간 복잡도를 계산하게 되며, O(1)이 된다. 재할당(할당->복사->해제)이 자주 발생하면 성능이 크게 떨어질 수 있으므로, 사전에 필요한 용량을 잘 파악하는 것이 중요하다.


#### References
- [분할상환분석](https://ko.wikipedia.org/wiki/%EB%B6%84%ED%95%A0%EC%83%81%ED%99%98%EB%B6%84%EC%84%9D)
- [What is Constant Amortized Time?](https://stackoverflow.com/questions/200384/what-is-constant-amortized-time)
- [분할 상환 분석(Amortized Analysis)](https://velog.io/@hysong/Algorithm-Amortized-Analysis-%EB%B6%84%ED%95%A0-%EC%83%81%ED%99%98-%EB%B6%84%EC%84%9D)
- [Amortized Time Complexity of Algorithms](https://medium.com/@satorusasozaki/amortized-time-in-the-time-complexity-of-an-algorithm-6dd9a5d38045#:~:text=Amortized%20time%20is%20the%20way,array%20and%20can%20be%20extended)



## #2
### 프리발즈 알고리즘(Freivalds' algorithm)
N*N 정사각행렬 A, B, C가 주어질 때, AB=C가 참인지, 거짓인지 반환하는 확률론적 알고리즘입니다. 단순 행렬곱 연산을 할 경우 시간 복잡도는 O(N<sup>3</sup>)이며, 빠르다고 알려진 Strassen 알고리즘의 시간 복잡도는 O(N<sup>log<sub>2</sub>7</sup>)이다. Freivalds' 알고리즘의 시간 복잡도는 O(kN<sup>2</sup>)이며, Freivalds 알고리즘이 틀릴 확률은 1/(2<sup>k</sup>)보다 작습니다. k는 연산을 시행하는 반복횟수를 의미하며, 사용되는 연산은 다음과 같습니다.
- 연산
    - 1. 크기가 n인 0 또는 1로만 구성된 값을 갖는 벡터 r을 랜덤하게 생성합니다.
    - 2.A(Br) = Cr인지를 확인하여, 결과가 같다면 AB=C가 맞다는 판정을 하고, 틀리면 AB=C가 아님을 판정합니다.
- 증명
    - [Freivalds' algorithm](https://en.wikipedia.org/wiki/Freivalds%27_algorithm)
- 코드
```python
import random
from typing import List
def freivald(a:List[List[int]], b:List[List[int]], c:List[List[int]]):
    """
    1. 0 또는 1로만 값이 채워진 길이가 N인 랜덤 벡터를 생성한다.
    2. A*(B*r) = C*r이 참인지 체크한다.
    """
    N = len(a)

    r = [random.randint(0, 1) for i in range(N)]

    def matrix_multiplication_by_random_vector(matrix:List[List[int]], r:List[int]):
        matrix_multiplication = [0] * N
        for row in range(N):
            for col in range(N):
                matrix_multiplication[row] = matrix_multiplication[row] + matrix[row][col] * r[col]
        return matrix_multiplication

    br = matrix_multiplication_by_random_vector(b, r)
    cr = matrix_multiplication_by_random_vector(c, r)
    abr = matrix_multiplication_by_random_vector(a, br)

    for abr_i, cr_i in zip(abr, cr):
        if abr_i - cr_i != 0:
            return False

    return True

def isProduct(a:List[List[int]], b:List[List[int]], c:List[List[int]], k:int):
    """k번만큼 freivald 알고리즘을 반복한다."""

    for i in range(0, k) :
        if (freivald(a, b, c) == False) :
            return False
    return True


```

#### References
- [카카오 추천팀 깃허브](https://github.com/kakao/recoteam/tree/master/programming_assignments/beale_ciphers)
- [Freivald’s Algorithm to check if a matrix is product of two](https://www.geeksforgeeks.org/freivalds-algorithm/)
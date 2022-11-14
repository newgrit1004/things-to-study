## Table of Contents

- [분할 상환 분석(Amortized Analysis)](#1)
- [밀러-라빈 소수 판별법](#2)
- [Quick Selection](#3)


## #1
### 분할 상환 분석
컴퓨터 공학에서 분할 상환 분석(amortized analysis)란 주어진 알고리즘의 시간 복잡도나 프로그램을 수행하는데 소요되는 시간 또는 메모리 같은 자원 사용량을 분석하기 위해서 사용하는 기법이다. amortized time의 의미는 특정 연산을 많이 반복하였을 때, 평균적으로 연산당 소요되는 시간을 의미한다. 특정 연산을 백만번 시도하였을 때 고려되는 요소는 해당 연산의 worst-case나 best-case가 아니라, 전체 소요된 시간을 신경쓰게 될 것이다. 따라서 해당 연산이 아주 가끔 느린 현상이 발생하는 것은 크게 신경쓰지 않는다. 예를 들어, dynamic array(동적 배열)에 반복적으로 새로운 item을 넣는 경우를 가정해본다. 일반적으로, 새로운 item을 추가하는 작업은 O(1)의 시간복잡도를 가진다. 하지만, array가 가득 찰 때마다 2배의 공간을 할당하고, 데이터를 새 영역에 복사하고, 이전 메모리를 free해주는 작업을 해야하며, 이러한 작업을 더블링(Doubling)이라 한다. 새로운 메모리 공간을 할당하고, 이전 메모리를 free해주는 작업은 현재 길이가 n인 array에 대해서 O(n)이 걸린다. Amortized Analysis를 통해 동적 배열에 데이터 추가하는 작업의 비용을 계산한다면 더블링이 발생하는 최악의 경우를 여러 번에 나눠서 시간 복잡도를 계산하게 되며, O(1)이 된다. 재할당(할당->복사->해제)이 자주 발생하면 성능이 크게 떨어질 수 있으므로, 사전에 필요한 용량을 잘 파악하는 것이 중요하다.


#### References
- [분할상환분석](https://ko.wikipedia.org/wiki/%EB%B6%84%ED%95%A0%EC%83%81%ED%99%98%EB%B6%84%EC%84%9D)
- [What is Constant Amortized Time?](https://stackoverflow.com/questions/200384/what-is-constant-amortized-time)
- [분할 상환 분석(Amortized Analysis)](https://velog.io/@hysong/Algorithm-Amortized-Analysis-%EB%B6%84%ED%95%A0-%EC%83%81%ED%99%98-%EB%B6%84%EC%84%9D)
- [Amortized Time Complexity of Algorithms](https://medium.com/@satorusasozaki/amortized-time-in-the-time-complexity-of-an-algorithm-6dd9a5d38045#:~:text=Amortized%20time%20is%20the%20way,array%20and%20can%20be%20extended)




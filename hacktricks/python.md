## Table of Contents

- [python anti-pattern](#1)
- [python good libraries](#2)

## #1
### python anti-pattern
anti-pattern이란 소프트웨어 개발에서 잘못된 방법이라고 여길 수 있는 패턴들을 의미합니다. anti-pattern과 반대되는 용어는 디자인 패턴입니다.anti-pattern이 포함된 코드가 야기할 수 있는 문제점들은 다음과 같습니다.
- 협업하기 힘든 코드
    - 읽기 힘든 코드
    - over-engineered된 코드
- 제공하는 서비스에서 하단과 같은 문제점이 발생할 수 있는 코드
    - 유지보수하기 힘든 코드
    - 퍼포먼스가 느린 코드
    - 불안정한 코드
    - 에러에 민감한 코드


매우 다양한 패턴의 python anti-pattern 들이 존재하기 때문에, 모든 케이스에 대해 소개할 수는 없지만, 하단의 Reference 링크를 통해 추가적으로 확인이 가능합니다.

- 빈 list, dict, tuple을 initialize할 때, literal syntax를 사용하지 않는 경우
```python
%timeit my_list_1 = []
#48.8 ns ± 9.55 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
%timeit my_list_1 = list()
129 ns ± 23.2 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
```
list literal('[]'), dictionary literal('{}'), tuple literal('()')을 이용하여 빈 container object를 초기화하는 것이 속도가 더 빠른 것을 볼 수 있습니다. 이러한 차이가 발생하는 이유를 파이썬의 dis 모듈을 통해 [바이트코드](https://docs.python.org/ko/3.8/glossary.html#term-bytecode)의 차이를 보면서 알아보겠습니다. dis 모듈은 주어진 파이썬 코드가 바이트코드 단위에서 어떻게 동작하는지 알 수 있는지 보여주는 모듈입니다.

```python
from dis import dis
dis("[]")
#   1           0 BUILD_LIST               0
#               2 RETURN_VALUE
dis("list()")
#   1           0 LOAD_NAME                0 (list)
#               2 CALL_FUNCTION            0
#               4 RETURN_VALUE
```
list literal의 동작은 단순한 것에 비해, list()의 동작은 list literal에 비해 좀더 길다는 것을 볼 수 있습니다. 파이썬 인터프리터는 '[]' 표현이 있을 때, 바로 list를 생성해야한다는 것(BUILD_LIST)을 알고 있습니다. 하지만, list()를 사용할 경우, 파이썬 인터프리터의 동작은 이름이 'list'인 객체를 찾은 뒤, list 함수를 호출하여 값을 리턴합니다. 인터프리터가 현재 존재하는 변수 중에 이름이 있는지 확인하는 순서는 "local scope -> enclosing scope -> global scope -> built-in scope"입니다. list는 built-in scope에 속하므로 인터프리터가 이름을 이용하여 찾는 것은 가능하지만 list literal에 비해 list를 생성하는 속도는 느릴 수 밖에 없습니다. 숫자와 string과 같이 인터프리터가 보자마자 바로 이해할 수 있는 값들을 literal value라 하는데, '[]', '{}', '()' 또한 literal value이므로 dis("[]")를 했을 때 위와 같은 결과가 나왔던 것입니다.


- generator expression을 사용해도 되는 상황에서 list/dict/set comprehension을 사용하는 경우
```python
#bad practice
concat_string = ''.join(['a', 'b', 'c'])

#good practice
cocnat_string = ''.join(('a', 'b', 'c'))
```
built-in 함수들은 generator expression을 input으로 주어도 잘 동작합니다. 특히 all()이나 any()는 파이썬에서 short-circuiting evaluation을 지원하는데, generator expression 대신 comprehension을 사용할 경우 퍼포먼스가 하락할 수 있습니다. short circuit evaluation이란 논리연산자 AND나 OR연산에서 결과가 확실할 경우 나머지 연산을 실행하지 않고 값을 리턴하는 경우를 의미합니다. AND 연산에서는 AND 앞 부분에서 False가 나올 경우, AND 뒷 부분 연산이 생략됩니다. OR 연산에서는 OR 앞 부분에서 True가 나올 경우 OR 뒷 부분 연산이 생략됩니다.

```python
import sys
test_list_comprehension = [i for i in range(10000)]
print(sys.getsizeof(test_list_comprehension))
#87616
test_generator_expression =(i for i in range(10000))
print(sys.getsizeof(test_generator_expression))
#112
```
comprehension과 달리 generator expression는 lazy evaluation(loading)이 가능하기 때문에 메모리 관리 측면에서 이점이 있습니다.


- 길이가 긴 list에서 element 존재 여부를 체크하는 경우
```python
from itertools import repeat, chain
def create_duplicate_list(lst:list, numbers:list):
    """Create a duplicate list.
    reference: https://stackoverflow.com/questions/41811099/create-duplicates-in-the-list"""
    return chain.from_iterable(repeat(i, j) for i, j in zip(lst, numbers))

lst = ['a', 'b', 'c']
numbers = [2, 4, 3]

duplicated_list = list(create_duplicate_list(lst, numbers))
duplicated_list
#['a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c']


#bad practice
check_a = "a" in duplicated_list

#good practice
check_a = "a" in set(duplicated_list)
```
길이가 긴 list에서 특정 element가 존재하는 것을 체크하고 싶을 경우에는 주어진 긴 list을 set으로 변환한 뒤 체크하는 것이 바람직합니다.


- 2개 이상의 string 객체를 concat 할 경우
```python
a = 'a'
b = 'b'
#bad practice
concat_string = a + ' ' + b
#good practice
concat_string = ' '.join((a,b))
```
2개 이상의 string 객체 2개를 concat해야할 경우, + 연산자보다는 join 연산자를 이용하는 것이 좋습니다. 반복문과 + 연산자를 사용하여 아주 많은 string 객체 리스트를 더한 결과를 리턴할 수도 있겠지만, 성능적으로는 join 연산이 훨씬 빠릅니다.


for loop과 + 연산자를 사용하여 주어진 list안에 들어있는 많은 갯수의 string을 다 합치는 경우 발생하는 로직은 다음과 같습니다. 파이썬 인터프리터는 주어진 string 사이에 띄어쓰기(whitespace)를 하기 위해서 모든 스트링마다 띄어쓰기와 스트링에 대한 메모리 할당이 이루어져야합니다.
하지만 join을 사용한다면 주어진 리스트의 첫번째 string의 공백에 대해서는 메모리 할당이 이루어질 필요가 없으며, 이외에는 + 연산과 같습니다. 그러므로 메모리 할당 횟수 차이에 의해 + 연산과 join 연산 사이의 성능 차이가 발생합니다.



- recursion의 속도를 빠르게 올리고 싶은 경우
```python
def fib_recursion(n:int):
    if n <= 1:
        return n
    else:
        return fib_recursion(n-1) + fib_recursion(n-2)


def fib():
    x1 = 0
    x2 = 1
    def get_next_number():
        nonlocal x1, x2
        x3 = x1 + x2
        x1, x2 = x2, x3
        return x3
    return get_next_number

def fib_closure(n:int):
    f = fib()
    for i in range(2, n+1):
        num = f()
    return num


from functools import lru_cache # if python version is less than python 3.9
#from functools import cache # if python version is equal or greater than python 3.9
@lru_cache()
def fib_recursion_with_lru_cache(n:int):
    if n <= 1:
        return n
    else:
        return fib_recursion(n-1) + fib_recursion(n-2)

#recursion, closure, recursion with lru_cache 비교
%timeit fib_recursion(35)
#5.88 s ± 770 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

%timeit fib_closure(35)
#10.1 µs ± 1.55 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)

%timeit fib_recursion_with_lru_cache(35)
#The slowest run took 9.50 times longer than the fastest. This could mean that an intermediate result is being cached.
#514 ns ± 567 ns per loop (mean ± std. dev. of 7 runs, 1 loop each)
```
재귀함수의 이용은 깊이가 깊어질 경우 계산량이 크게 증가하게 됩니다. 재귀함수와 관련된 대표적인 예시 문제는 피보나치 문제가 있습니다. 주어진 예시로는 n을 35로 설정하였습니다. 이것을 식으로 나타낸다면 fib(35) = fib(34) + fib(33) = (fib(33)+fib(32)) + fib(33) ... 이 될 것입니다. 재귀함수를 이용하게 되면 이미 계산한 결과를 이용하지 못하고 다시 계산하는 문제점이 있습니다. 해결하는 방법은 이미 계산된 값이 존재하면 계산된 값을 사용하도록 하는 cache를 생성하는 것입니다. 이와 같은 케이스를 일반적으로는 다이나믹 프로그래밍과 같은 방법으로 해결하지만, [lru_cache](https://docs.python.org/ko/3/library/functools.html)(Least Recently Used)와 같은 데코레이터를 재귀함수에 붙여줌으로써 해결하는 방법도 가능합니다. lru cache를 사용하는 경우는 해당 함수 내에서 결과값만 중요할 때 사용합니다.



#### References
- [18 Common Python Anti-Patterns I Wish I Had Known Before](https://medium.com/towards-data-science/18-common-python-anti-patterns-i-wish-i-had-known-before-44d983805f0f)
- [The Little Book of Python Anti-Patterns](https://docs.quantifiedcode.com/python-anti-patterns/index.html)
- [No, [] And list() Are Different In Python](https://towardsdatascience.com/no-and-list-are-different-in-python-8940530168b0)

- [[Algorithm] Short Circuit Evaluation이란?](https://twpower.github.io/53-about-short-circuit-evaluation)
- [Do Not Use “+” to Join Strings in Python](https://medium.com/towards-data-science/do-not-use-to-join-strings-in-python-f89908307273)
- [The Little Book of Python Anti-Patterns](https://github.com/quantifiedcode/python-anti-patterns)




## #2
### python good libraries
- 생산성을 높이는 오픈 소스 라이브러리
    - 리팩토링
        - [refurb](https://github.com/dosisod/refurb); 작성한 코드 중에 고칠 수 있는 부분들을 알려주는 라이브러리
    - 디버깅
        - [icecream](https://github.com/gruns/icecream) : print()나 log()로 디버깅하는 것 대신, icecream을 이용한 디버깅이 여러 측면에서 좋음
    - 함수형 프로그래밍
        - [more-itertools](https://github.com/more-itertools/more-itertools) : python의 내장 라이브러리인 itertools를 좀더 직관적으로 사용할 수 있게 도와주는 라이브러리



#### References
- [Life is short, you need python](https://awesome-python.com/)



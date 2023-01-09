## Table of Contents

- [JAX](#1)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](#2)
- [AMP(Automatic Mixed Precision)](#3)
- [모델 경량화](#4)
- [분산 딥러닝](#5)


## #1
### JAX
JAX는 [구글](https://github.com/google/jax)이 개발한 파이썬 라이브러리로 numpy.array를 CPU/GPU/TPU에서 연산이 가능하도록 하여 수치 계산 실행 속도를 크게 높였다. 2020년 DeepMind에서 머신러닝 성능 최적화 및 가속화를 위해 JAX를 사용하고 있다. JAX를 기반으로 만든 뉴럴 네트워크 프레임워크로는 Flax가 있다.
Flax는 텐서플로우나 파이토치와 같은 뉴럴 네트워크 프레임워크 중 하나이며 [구글](https://github.com/google/flax)이 개발하는 고성능을 위한 머신러닝 프레임워크이다. [stable diffusion 모델을 JAX와 TPU를 이용하여 inference할 경우, 8초 안에 8개의 이미지를 생성할 수 있다고 한다.](https://twitter.com/psuraj28/status/1580640841583902720?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1580640841583902720%7Ctwgr%5E3b3a0f2f32cabf71fa3fa3c22849dfd0b632b13d%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fwww.redditmedia.com%2Fmediaembed%2Fy3889p%3Fresponsive%3Dtrueis_nightmode%3Dfalse) JAX 라이브러리 자체는 플랫폼에 독립적이기 때문에, 기존 tensorflow나 pytorch에서 numpy.array를 GPU/TPU 연산으로 가속화하고 싶은 경우 조합해서 사용하는 것 또한 가능하다.


```python
import numpy as np
import jax.numpy as jnp
import jax

def f(x):  # function we're benchmarking (works in both NumPy & JAX)
    return x.T @ (x - x.mean(axis=0))

x_np = np.ones((1000, 1000), dtype=np.float32)  # same as JAX default dtype
%timeit f(x_np)  # measure NumPy runtime -> 16.2 ms per evaluation on the CPU

%time x_jax = jax.device_put(x_np)  # measure JAX device transfer time -> 1.26 ms to copy the numpy arrays onto the GPU
f_jit = jax.jit(f)
%time f_jit(x_jax).block_until_ready()  # measure JAX compilation time -> 193 ms to compile the function
%timeit f_jit(x_jax).block_until_ready()  # measure JAX runtime -> 485 microsecond per evaluation on the GPU
```
함수가 컴파일되고 나면, GPU 위에서 JAX 기반 함수가 실행이 될 때 numpy보다 30배가 더 빠름을 확인할 수 있다. JAX 라이브러리 FAQ에서 비교한 결과로, 전체 애플리케이션을 실행할 때 성능 평가는 데이터 전송(data transfer) 및 컴파일(compilation)이 포함되게 된다. 충분히 큰 데이터를 사용하여 비교할 때 JAX의 컴파일 후 성능적 우위가 더 강조가 된다고 보여진다. 10*10과 같은 작은 사이즈의 데이터를 이용하여 비교할 경우, JAX/GPU가 NumPy/CPU보다 10배 느리다고 한다(100µs vs 10µs).


JAX는 [XLA 컴파일러](https://www.tensorflow.org/xla) 위에서 JIT(Just In Time) 방식으로 함수를 컴파일하기 때문에 속도가 매우 빠르다. XLA 컴파일러는 Tensorflow용 컴파일러로 JAX를 사용할 경우, 코드를 분석하고 GPU/TPU에서 실행되기 위한 최적화방법을 결정하고 GPU/TPU 코드를 생성하여 계산을 수행한다.

XLA 컴파일러가 코드 최적화를 하는 방법 중 하나는 JIT(Just-In-Time) compilation이다. JIT 컴파일은 런타임 시 코드를 컴파일해서 프로그램을 실행하는 방식이다. JAX에서는 특정 함수에 @jit 데코레이터를 붙여줌으로써 해당 함수를 JIT 컴파일할 수 있다. [XLA를 사용할 경우, BERT 트레이닝 속도 향상이 최대 7배까지 증가하였으며, GPU당 배치사이즈를 5배 증가시킬 수 있었다. 메모리 사용량이 줄어들면서, gradient accumulation과 같은 advanced technique도 사용이 가능했다.](https://blog.tensorflow.org/2020/07/tensorflow-2-mlperf-submissions.html)


JAX 라이브러리의 대표적인 api는 jit, grad, vmap 등이 있다.
jax.grad 함수는 하나 이상의 인풋에 대한 함수의 그라디언트를 계산하는데 사용된다.
```python
import jax.numpy as jnp
import jax

def sum_of_squares(x:float, y:float)->float:
    return x**2 + y**2

grad_sum_of_squares = jax.grad(sum_of_squares)
x, y = 1.5, 2.5
grad = grad_sum_of_squares(x, y)
print(grad)  # Output: (3.0, 5.0)
```

jax.vmap 함수는 함수를 벡터화하거나 주어진 인풋 집합에 병렬로 함수를 적용할 수 있는 함수이다. 벡터화(Vectorization)은 여러 개의 인풋 데이터에 함수를 동시에 적용하는 방식을 뜻한다. 벡터화를 사용하면 함수를 각 인풋에 별도로 적용하는 것보다 훨씬 빠르게 처리하는 것이 가능하다.
```python
import jax.numpy as jnp
import jax

def mul(x, y):
    return x*y

x = jnp.array([1.0, 2.0])
y = jnp.array([4.0, 5.0])

vmul = jax.vmap(mul)

z = vmul(x, y)
print(z)  # Output: [4.0, 10.0]
```


#### References
- [Google Jax 와 Flax Library](https://mjshin.tistory.com/14)
- [Benchmarking JAX code](https://jax.readthedocs.io/en/latest/faq.html#benchmarking-jax-code)
- [ML 최적화 1. JIT & google JAX](https://brunch.co.kr/@chris-song/99)
- ["가속기를 단 넘파이" 구글 JAX 시작하기](https://www.itworld.co.kr/news/245590)
- [[파이썬] JAX Quick Start / JAX란?](https://koreapy.tistory.com/1022)


## #2
### Bag of Tricks for Image Classification with Convolutional Neural Networks
CNN 기반 딥러닝 모델 아키텍처를 많이 바꾸지 않고도 사소한 트릭(e.g. 손실 함수, 데이터 전처리, 최적화 등)들을 적절히 모아 사용한다면 큰 성능 향상을 이룰 수 있다는 점을 보여준 논문이다. 기존 아키텍처에서 다양한 트릭들을 섞었기 때문에 FLOPs(FLoating point Operations)가 증가하지만, top-1 및 top-5 정확도가 크게 향상된 것을 논문의 결과 부분에서 확인할 수 있다.

- 이미지 데이터에 대한 베이스라인 및 논문 방법 비교
    - 기존에 일반적으로 쓰이던 방식
        - 실험 세팅
            - 8개의 Nvidia V100 GPU
            - Xavier 가중치 초기화 알고리즘
            - batch size는 256
            - 전체 epoch는 120으로 설정하고, 30번째 epoch마다 learning rate를 10으로 나눔
            ```python
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            ```
        - 학습 방식
            - (1) 이미지를 불러온 뒤, 32bit floating으로 디코딩을 수행한다.
            - (2) 이미지 내에서 랜덤하게 직사각형 영역을 잘라내서 224*224 크기로 변경하여 항상 고정된 크기의 인풋값으로 뉴럴 네트워크에 이미지가 들어갈 수 있도록 한다.
            - (3) 이후 50% 확률로 좌우 반전하거나 또는 색조/채도/밝기 등을 랜덤하게 변경함으로써 data augmentation을 진행한다.
            - (4) 최종적으로 입력 데이터를 정규화하여 정규화가 수행된 이미지 데이터 형태를 유지한 상태로 뉴럴 네트워크에 입력값으로 넣습니다.
    - 논문에서 제안한 방법
        - Large Batch Training
            - 일반적으로 주어진 GPU 자원이 허용하는 내에서 가능한 큰 크기의 batch size를 설정한다. 하지만, batch size가 클 경우, convergence rate가 감소할 수 있다. 동일한 횟수로 학습한다면 batch size가 큰 모델이 성능이 낮을 수 있다.
            - 논문에서는 batch size를 크게 설정하면 gradient가 상대적으로 덜 noisy하기 때문에 learning rate를 크게 설정해도 괜찮다는 점이 있다고 한다. 그래서 초기 학습률을 0.1 * batch_size / 256 으로 설정하도록 한다.
            - 너무 큰 learning rate를 초반부터 적용할 경우 초반부 학습 과정이 불안정할 수 있다.  그래서 워밍업(warmup)을 통해 초반 학습 단계에서 특정 스텝까지는 선형적으로 learning rate를 증가시킨다. 초기 m번의 배치에 대해서 학습률을 초기 학습률(0.1*batch_size/256) 나누기 m으로 설정합니다.
            - 일반적으로 residual block의 뒷 부분에 batch normalization이 사용되며 gamma와 beta를 1과 0으로 초기화한다. 논문에서는 학습을 시작할 때 gamma를 0으로 초기화하여 초반 학습 난이도를 쉽게 만들 수 있다고 주장한다. residual block에서 gamma를 0으로 설정한다는 것은 residual block의 영향력을 0으로 만드는 행위와 같다. 초기 학습 단계에서 입력 값이 그대로 identity mapping으로 전달되게 만들고, gamma값을 서서히 업데이트하면서 residual block의 결과가 점차 학습 결과에 영향을 미칠 수 있는 구조로 학습한다.
        - Low-Precision Training
            - 일반적으로 뉴럴 네트워크는 32bit floating point 정밀도(precision)를 사용한다.
            - 다양한 하드웨어는 더 낮은 정밀도 자료형을 효율적으로 지원하는데, 예를 들어 NVIDIA V100은 FP32에 대해서는 14TFLOPS(테라플옵스), FP16에 대해서는 100TFLOPS 이상의 성능을 제공한다.
            - 정밀도가 낮아져서 성능 저하가 발생할 수 있지만, 다른 트릭들을 통해 전반적인 성능을 더 올릴 수 있다.
        - Model Tweaks
            - 4번의 스테이지에 거쳐 각 스테이지마다 downsampling을 진행한 뒤 residual block을 통과시킨다.
            - stride는 downsampling 목적으로도 사용되는데, 기본으로 사용되는 stride size는 1을 사용한다.
            - 기존 downsampling block에서는 첫 번째 Convolution layer에서 1 * 1 512개의 커널, stride 2를 사용했었는데, 정보의 큰 손실을 없애기 위해 첫 번째 convolution layer에서 stride 1로 설정하고 그 다음 layer에서 stride 2로 설정하는 방식을 사용했다. 그리고 downsampling의 오른쪽 부분에서는 Convolution layer 1*1, stride=2를 사용하면 3/4의 정보가 손실되기 때문에 정보 손실을 줄이기 위해 average pooling layer로 변경한다.
        - Cosine Learning Rate Decay
            - 일반적으로 쓰는 learning rate decay는 step decay라 부르며, N번째 에포크마다 learning rate를 확 떨어뜨리는 방식이다. Cosine decay는 1/2 * (1+cos(t*pi/T)) * 초기 learning_rate이다. 여기서 T는 전체 배치의 개수를 의미하고, t는 각 배치를 의미한다. step decay는 learning rate가 감소할 때마다 accuracy가 급격하게 올라가는 것을 볼 수 있고, cosine decay는 accuracy가 천천히 상승한다.
        - Label Smoothing
            - 일반화 성능을 높이기 위해 사용하는 방법으로, 정답 레이블에 대해서 100% 확률을 부여하지 않고, 입실론을 뺀 값 만큼 확률을 부여한다. 그리고 나머지 레이블에 대해서 균등하게 확률을 부여한다.
        - Knowledge Distillation
            - 티처 모델이 존재하는 상황에서 티처 모델의 아웃풋을 이용해서 스튜던트 모델이 학습하는 방식이다. 논문에서는 Teacher model로 ResNet-50보다 큰 ResNet-152를 사용하였고, 이를 이용하여 ResNet-50을 학습시켰다.
        - Mixup Training
            - 학습을 진행할 때 두 개의 샘플씩 묶어서 뽑은 데이터를 학습에 사용한다. 이 때 레이블 값은 믹싱에 사용한 값의 비율을 레이블 값으로 준다.



    - 성능 분석
        - Baseline
            - ResNet-50 with FP32, batch_size = 256
        - Efficient
            - ResNet-50 with FP16, batch_size = 1024
            - Baseline과 비교하여 모델 학습속도 2배~3배까지 증가
            - 정확도 또한 BaseLine에 비해 우위에 있음




#### References
- [CVPR2019 논문 출처](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks (CVPR 2019) 꼼꼼한 딥러닝 논문 리뷰](https://www.youtube.com/watch?v=2yxsg_aMxz0)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks Review](https://hoya012.github.io/blog/Bag-of-Tricks-for-Image-Classification-with-Convolutional-Neural-Networks-Review/)
- [CNN 꿀팁 모음 (Bag of Tricks for Image Classification with Convolutional Neural Networks) 논문 리뷰](https://phil-baek.tistory.com/entry/CNN-%EA%BF%80%ED%8C%81-Bag-of-Tricks-for-Image-Classification-with-Convolutional-Neural-Networks-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)



## #3
### AMP(Automatic Mixed Precision)
일반적으로 딥러닝에서 소수점을 처리할 때 32비트(FP32)를 사용하여 처리한다. 더 많은 비트를 사용할수록 정확한 연산이 가능하고, 더 적은 비트를 사용할수록 빠른 연산이 가능하다. FP32로 모델 학습을 진행하면 loss 계산, 그라디언트 계산 등이 무거워진다. 무거운 모델인 경우 단일 GPU로 학습하기 어려울 수도 있다. FP16으로 모델 학습을 진행하면 FP32 대비 절반의 메모리만 필요하며 이에 따라 배치 사이즈를 크게 하여 빠른 학습이 가능하다. 하지만 FP16은 표현할 수 있는 숫자의 범위가 매우 작다. FP16에서 1.0001은 1로 계산되는 문제가 있다.

Mixed Precision은 대부분의 경우에 FP16을 사용하고 특정 경우에만 FP32를 사용해서 학습하는 방식이다. FP32에서 mixed precision을 사용한 경우 학습 속도 향상은 3~5배까지 증가하며, 성능에 대한 손실은 거의 없다.
- 수치적 안전 및 성능 향상 : Convolution, Matmul -> FP16
- 수치적 중립 : Max, Min -> 상황별 적용
- 조건부로 수치적 안전 : Activations -> 상황별 적용
- 수치적 위험 : Exp, Log, Pow, Softmax, Reduction Sum, Mean -> FP32

```python
#default(FP32)
net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        output = net(input)
        loss = loss_fn(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Default precision:")


#Autocast and GradScaler
"""
Autocast: 네트워크에서의 loss 계산이나 forward pass에서만 사용하는 것을 권고함.
GradScaler:FP16의 경우 너무 작은 값이 0이 될 수가 있어서 스케일링해서 크게 만들기도 함.
"""
use_amp = True

net = make_model(in_size, out_size, num_layers)
opt = torch.optim.SGD(net.parameters(), lr=0.001)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

start_timer()
for epoch in range(epochs):
    for input, target in zip(data, targets):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = net(input)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad() # set_to_none=True here can modestly improve performance
end_timer_and_print("Mixed precision:")

```


FP32를 FP16으로 변환한 뒤에 순전파 및 역전파를 FP16 자료형으로 수행합니다.
가중치 업데이트를 위해 다시 FP32로 변환합니다.
FP32만 사용했을 때보다 이미지 분류 작업에서 유사하거나 더 좋은 성능을 보일 수 있습니다.

사용하는 비트의 수를 절반으로 바꾼 뒤, 포워드와 백워드 과정을 통해 실제 그래디언트를 구한다. 구한 그래디언트를 이용해서 가중치를 업데이트하고, 이 과정에서는 다시 float 32비트를 사용한다.
학습 속도를 매우 빠르게 만들었음에도 불구하고 성능은 유사하거나 더 좋게도 가능.



#### References
- [PyTorch AMP - 1 : Mixed Precision Training 논문 리뷰](https://computing-jhson.tistory.com/36)
- [Automatic Mixed Precision (AMP)](https://otzslayer.github.io/ml/2022/01/31/automatic-mixed-precision.html)
- [AUTOMATIC MIXED PRECISION](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)


## #4
### 모델 경량화 및 추론 속도 향상
모델 경량화의 장점은 연산 비용 감소 및 성능 증가이다. 비용 감소의 예시로는 같은 작업을 더 적은 서버의 수로 해도 된다거나, 전력 소모를 줄일 수 있다는 점이 있다. 성능이 증가한다는 점은 네트워크 트래픽을 감소시키며, 주어진 작업을 빠르게 처리하므로 초당 작업량이 늘어난다. <b>경량화된 모델은 하드웨어에서 실행되기 때문에 타겟 디바이스(e.g. GPU/엣지 디바이스)가 어느 정도를 지원하는지를 파악하는 것이 중요하다.</b>

경량화 방법은 크게 Qunatization과 Pruning 방법이 존재한다. Pruning 방법의 경우, <b>The Lottery Ticket Hypothesis</b>라는 논문에서 다수의 weight를 pruning 하더라도 성능이 잘 나온다는 점을 보여주었다. 하지만 단순하게 Pruning을 적용하고 사용할 경우, weight에 0이 많이 존재하는 모델에 불과하여 속도 및 용량이 그대로다. 모델 weight에 0이 많이 들어있을 경우, 모델 압축 효율이 좋다는 장점은 있다. 그리고 Transformer 모델의 경우, 아직(2021년)은 하드웨어 및 소프트웨어가 CNN 연산에 최적화가 많이 되어있어서 좀 더 지켜보는 것이 좋다.


- CNN Layer 압축
    - Tucker Decomposition이라는 방법을 통해 Convolution Layer Decomposition 하면 파라미터의 수를 획기적으로 줄일 수 있다.(파라미터 수 88% 감소, 성능 매우 안좋음)
        - The number of parameters
            - 128 * 256 * 3 *3 Convolution : 294,912
            - Tucker Decomposition : (128 * 64 * 1 * 1)+(64 * 32 * 3 * 3)+(32 * 256 * 1 * 1) = 34,816
        - 정보 손실이 너무 커서 정확도가 안좋다.
    - 랜덤 텐서를 1개 만든 뒤, 원래 convolution layer와 decomposition convolution layer에 통과시켜서 오차가 적은 layer에 대해서만 사용(파라미터 수 40% 감소, 성능 1% 저하)
    - 원래 Convolution  layer에 Pruning을 먼저 단순하게 적용하여 weight 내부에 0이 많도록 한 뒤, Tucker Decomposition 진행(파라미터 수 5% 추가 감소)
    - 모든 layer마다 pruning ratio를 다르게 결정(binary search 이용)한 후 Tucker Decomposition 진행(파라미터 수 2% 추가 감소)


- 추론 속도 향상
    - GPU에서 FP16을 지원한다면 Half Precision 사용 추천
    - CPU/GPU 간에 메모리 이동을 지양할 것
    - TensorRT가 사용 가능하다면 사용하고, 그 외에는 libtorch c++을 사용할 것
    - Layer별로 속도 profiling을 하여 병목 현상 생기는 지점을 체크할 것


#### References
- [[MODUCON 2021] 실용적인 딥러닝 모델 경량화 & 최적화를 해보았습니다-임종국[AI + X]](https://www.youtube.com/watch?v=QZekRr4xUAk)
- [The Lottery Ticket Hypothesis Finding Sparse, Trainable Neural Networks 논문 리뷰](https://ysbsb.github.io/pruning/2020/04/21/Lottery-ticket-hypothesis.html)
- [kindle](https://github.com/JeiKeiLim/kindle)
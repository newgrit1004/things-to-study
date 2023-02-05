## Table of Contents

- [GPU Profiling](#1)
- [Scalene](#2)


## #1
### GPU Profiling
GPU Profiling이란 특정 작업이나 애플리케이션을 실행하는 동안 GPU 성능을 측정하고 분석하는 프로세스를 말한다. GPU 프로파일링에는 GPU 메모리 사용량, GPU Utilization 등 다양한 성능 메트릭을 측정하여 병목 현상을 파악하고, 파악한 내용을 바탕으로 GPU 성능을 최적화하는 작업을 진행한다.


- GPU 메모리
    - GPU 메모리는 학습 및 추론에 필요한 데이터와 중간 결과를 저장하는데 사용된다.
    - 전용 메모리(dedicated memory)와 공유 메모리(shared memory)로 나뉘며 전용 메모리는 GPU 전용으로 사용되고 공유 메모리는 GPU, CPU 모두 엑세스가 가능하다.
    - 일반적으로 딥러닝 모델 트레이닝할 때 GPU 메모리가 많이 요구되고 인퍼런스할 때는 훨씬 적게 요구되거나 또는 CPU만 사용하여 인퍼런스 하므로 상관없는 경우도 있다.
    - 트레이닝 때 요구되는 GPU 메모리
        - GPU memory = 배치 사이즈 * (모델 가중치 메모리 + 활성 함수 메모리 + 피처 맵 저장 메모리 + 그라디언트 저장 메모리)
        - 배치 사이즈 : 1번의 포워드 패스와 백워드 패스에서 처리되는 샘플의 개수
        - 모델 가중치 메모리(weights) : 모델의 파라미터를 저장하기 위한 메모리
        - 활성 함수 메모리(activation functions) : 모델 아키텍처에 들어있는 활성 함수를 저장하기 위해 사용되는 메모리
        - 그라디언트 메모리(gradients) : 백워드 패스 동안 저장되어야하는 그라디언트 메모리
        - 피처 맵 메모리 (feature maps) : 배치 사이즈 * 채널 갯수 * Height * Width * 데이터 타입별 사이즈로 계산됨
    - 파이토치 메모리
        - 파이토치에서는 caching memory allocator를 사용하여 메모리 할당을 빠르게 할 수 있다.
        - nvidia-smi에서는 caching memory allocator가 사용하고 있는 메모리를 사용중인 메모리로 표기하는데, 실제로 사용되지 않는 메모리도 있다.
        - torch.cuda.memory_allocated(), torch.cuda.max_memory_allocateD() 등 [파이토치 api](https://pytorch.org/docs/stable/notes/cuda.html?highlight=buffer#memory-management)를 통해 더 자세한 내용을 확인할 수 있다.


- GPU Utilization
    - 이전 샘플 기간(제품별로 1/6초 ~ 1초 사이)동안 하나 이상의 커널이 GPU에서 실행된 시간의 백분율
    - GPU Utilization이 낮다는 것은 GPU를 제대로 활용하고 있지 못한다는 의미이다.


GPU 프로파일링을 할 수 있는 도구 중 하나로는 NVIDIA에서 제공하는 Nsight Systems과 NVTX의 조합을 이용할 수 있다. NVTX는 파이토치에서 기능을 제공하므로 import해서 사용하는 것이 가능하다.

- nvtx 예시 코드
```python
from torch.cuda import nvtx
nvtx.range_push('data loading')
batch = next(dataloader)
nvtx.range_pop()
```

nvtx의 사용 방법의 예시로는 파이토치의 데이터로더(DataLoader)가 있다. 파이토치의 데이터로더에서 fetch라는 함수가 사용되는 시간을 측정할 수 있으면 데이터 로딩에서 사용되는 시간을 알 수 있다. 몽키 패칭이라는 방법을 통해 이 과정을 다른 파이토치 메소드에 대해서도 쉽게 적용하는 것이 가능하다. 몽키 패칭이란 런타임에 클래스나 모듈의 함수를 수정하는 행위를 말한다. 따라서 파이토치 깃 레포지토리를 fork한 후 특정 메소드에만 nvtx marker를 다는 등의 작업을 할 필요가 없다.

- 몽키패칭을 이용한 nvtx marker를 다는 예시 코드
```python
from torch.cuda import nvtx
def monkey_patch(mod, func_name):
    func = getattr(mod, func_name)
    msg = f'<{func_name} in <{mod.__name__}>'
    def wrapper_func(*args, **kwargs):
        nvtx.range_push(msg)
        result = func(*args, **kwargs)
        nvtx.range_pop()
        return result
    setattr(mod, func_name, wrapper_func)
```


- GPU Profiling을 통한 성능 개선 예시
- 초기 세팅
    - 모델 : ResNet50
    - GPU : V100 1개
    - Mixed precision 사용됨
    - Metric
        - GPU Utilization : 50%
        - Throughput : 450 imgs/sec
- 프로파일링 결과
    - GPU 트레이닝 시간 : 배치 1개당 350ms
    - CPU 전처리 시간 : 배치 1개당 1500ms
    - 피드백
        - worker를 2개에서 5개로 변경 후 GPU Util 90%, Throughput 690imgs/sec 로 상승
- 추가 성능 개선을 위한 디테일한 개선
    - Memory Format
        - 이미지 데이터는 일반적으로 4-D tensor 데이터
        - NHWC 데이터를 channels last memory fromat이라 부르고,
        NCHW 데이터를 contiguous memory format(channels first memory format)이라 부른다.
        - [cuDNN에서 convolutional algorithm이 여러 종류가 존재하는데, mixed-precision을 사용하고 있으므로 Tensor Cores를 활용하고 있는 알고리즘은 channels last memory format(NHWC)를 요구한다. NCHW 메모리 포맷의 데이터는 사용가능하나 transpose opeartion이 자동으로 진행되므로 오버헤드가 발생할 수 있다.](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html#tensor-layout)
        - 메모리 포맷을 NCHW에서 NHWC로 변경한 결과, GPU 프로파일링 결과에서 nchwTonhwcKernel이 더이상 보이지 않으며, GPU Util은 90에서 68%로 떨어졌고, throughput은 690에서 850으로 증가하였다. GPU training 시간 또한 350ms에서 230ms로 감소되었으므로 worker 개수를 5개에서 7개로 늘리면 다시 GPU Util이 68%에서 83%로 증가하며, Throughput은 950으로 증가하게 되었다.
    - Memory Pinning
        - GPU가 30ms 쉬는 동안, 'Memcpy HtoD'라는 CPU 영역에서 GPU 영역으로 메모리 카피하는 동작을 진행하는 것을 확인하였다.
        - 호스트에서 호스트로 메모리 카피가 이뤄지는 것이기 때문에 사실 GPU가 쉴 이유는 없다.
        - pinned memory buffer는 파이토치 텐서들이나 스토리지에 핀 메모리라는 함수가 있는데, 이 함수를 호출하게 되면 데이터를 pagable 영역에서 pin 영역으로 옮기게 된다. 핀 영역으로 옮긴 데이터에 대해서 메모리 카피가 일어날 것을 파이토치는 기대를 하고 있다. 예상하지 못하게 pagable 영역의 텐서에 대해서 to나 cuda() 명령어로 메모리 카피를 수행하게 되면 내부적으로 pin 영역으로 옮기면서 시간이 좀더 소요된다.
        - 파이토치 데이터 로더를 만들 때 pin-memory 옵션을 주는 걸로 해결 가능하다.
        - 이 작업을 완료하면 Memcpy는 30ms -> 4ms로 감소하며, pin-memory와 관련된 쓰레드가 따로 생겨서 동작하는 것이 체크가능하다. GPU Util은 83->94%로 증가하였으며 throughput은 950->1070 imgs/sec로 증가하였다.
    - [CUDA Streams](https://pytorch.org/docs/stable/notes/cuda.html
)
        - Stream이라는 개념은 쓰레드와 비슷하게 생각하면 되며, Stream 안에서 일어나는 operation은 항상 시리얼하게 동작하게 된다. 따라서, 1개의 스트림에서 트레이닝과 메모리카피가 같이 존재한다면, 메모리 카피가 끝날 때까지는 다음 트레이닝이 일어날 수가 없다.
        - 새로운 스트림을 만들어서 1번째 스트림에서는 트레이닝을 진행하고 2번째 스트림에서는 Memcpy를 진행하는 것이 가능하다.
        - 하지만 비동기적 Memcpy를 진행하면 여기서 발생하는 문제는 pytorch가 충분히 관리해주지 못한다.
            - 메모리 관리 X
            - 각 스트림의 operation의 실행 순서 보장 X
            - 발표자의 경우에는 데이터로더 객체 바깥에서 제너레이터를 하나 생성하고, 해당 제너레이터는 memcpy 1개를 수행하고 이전 memcpy했던 객체를 yield하는 방식을 사용한다.
            - GPU Utilization 94 -> 96%, throughput 1070 -> 1090 imgs/sec


#### References
- [Pytorch Memory management](https://pytorch.org/docs/stable/notes/cuda.html?highlight=buffer#memory-management)
- [Useful nvidia-smi Queries](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries)
- [[Techtonic 2021] GPU profiling을 통한 deep learning 학습 최적화 - 김성준 프로](https://www.youtube.com/watch?v=bbrzuKgqgdc)
- [[Profile] GPU profile을 통한 병목 진단 및 개선](https://pajamacoder.tistory.com/11)
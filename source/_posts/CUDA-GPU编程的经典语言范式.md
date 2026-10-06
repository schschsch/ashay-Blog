---
title: CUDA:GPU编程
date: 2026-10-01 20:38:50
tags:
    - 算子开发
    - AI Infra
    - GPU架构
cover: /figs/Blog3.png
---

# CUDA编程

CUDA是NVIDIA提供的GPU专用的编程语言，是将GPU从图形学相关计算专用设备推广到通用计算平台的关键桥梁，虽然Triton已经大幅度简化了算子编程的难度和开发周期，并且LLM在这方面的能力也已经达到了Expert的境界，但无论如何，通过CUDA了解Nvidia GPU的硬件底层架构，总是有用的。

> 还记得大二时，第一个SRTP项目就是CUDA加速的GNSS信号软件接收机的算子开发。
> 虽然当时做的工作实际上就是最简单的AllReduce优化，但是在当时Vibe Coding还只是初露头角，学习起来还是挺费劲的。
> 没想到现在LLM大人已经能把算子开发工作做的炉火纯青了，真是让人唏嘘啊。

## CUDA编程模型

### 函数
CUDA中存在以下几种函数描述符，从而适配了CUDA编程的异构需求：部分代码在CPU上运行，部分代码在GPU上运行，并且需要一个从CPU到GPU的桥梁，也就是接口。
```c++
__global__ void func();
__device__ int func();
__host__ int func();
```
- "\_\_host\_\_"描述符用于显式声明这个函数在CPU（主机端）上执行。由于CUDA中不加描述符的函数默认就是主机函数，所以单独写一个"\_\_host\_\_"确实比较少见，但它并不是没有意义：当它与"\_\_device\_\_"同时使用时（`__host__ __device__`），编译器会为同一个函数同时生成主机端和设备端两个版本，让同一份逻辑在两侧复用，这是它真正有价值的用法。
- "\_\_global\_\_"描述符就是GPU的调用接口，它表示这个函数由CPU端调用，在GPU端运行，同时调用方式也与传统C/C++函数不同。
- "\_\_device\_\_"描述符如字面意思一样，在GPU上运行，同时只能被GPU上执行的代码调用。

一段经典的CUDA代码可以是这样的:
```c++
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// GPU 上的辅助计算函数
__device__ float myCos(float x) {
    float x2 = x * x;
    return 1.0f - x2 / 2.0f + x2 * x2 / 24.0f;
}

// 入口函数，被 CPU 上的代码调用
__global__ void vecCos(float* input, float* output, int len) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = id; i < len; i += stride) {
        output[i] = myCos(input[i]);
    }
}

int main() {
    const int len = 1 << 20;  // 1M 个元素
    const size_t bytes = len * sizeof(float);

    // 主机端内存
    float* h_in = (float*)std::malloc(bytes);
    float* h_out = (float*)std::malloc(bytes);
    if (h_in == nullptr || h_out == nullptr) {
        std::fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < len; ++i) {
        h_in[i] = -1.0f + 2.0f * i / (len - 1);
    }

    // 设备端内存
    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc((void**)&d_in, bytes);
    cudaMalloc((void**)&d_out, bytes);

    // 主机 -> 设备
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // 启动 kernel
    int block = 256;
    int grid = 1024; 
    vecCos<<<grid, block>>>(d_in, d_out, len);

    cudaGetLastError();
    cudaDeviceSynchronize();

    // 设备 -> 主机
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    // 释放资源
    cudaFree(d_in);
    cudaFree(d_out);
    std::free(h_in);
    std::free(h_out);

    return 0;
}
```
第一次间到CUDA繁琐的操作逻辑想必是望而却步，但实际上，现在只需要关系函数调用栈是怎样的即可。
main函数调用vecCos后，vecCos又调用myCos。GPU侧计算的同时main函数（也就是说，CPU侧），在调用后立刻返回，异步的完成计算。

### 核函数调用

vecCos<<<grid,block>>>(d_in,d_out,len)中的特殊调用格式，就是"\_\_global\_\_"描述的函数，即核函数的调用方式。

核函数调用自然是不会让CPU陷入等待的，我们可以一次性调用多个核函数使GPU能够不断接受指令，使其计算性能得到充分利用（实际充分与否取决于核函数计算量与传输数据量的比值）。

### Block、Grid参数

核函数调用时需要指定Grid和Block两个参数，两者共同决定了调用多少个计算单元，并且会在核函数内部隐式的产生四个变量：gridDim、blockIdx、blockDim、threadIdx。从而使得核函数虽然在所有被调用的计算单元内都执行同一段代码，但能够通过blockIdx、threadIdx这两个索引来手动分配不同thread分工合作。

核函数调用时指定的Block、Grid可以是一维、二维、三维的，blockDim和gridDim就是它们在核函数内部的体现，它们两两对应并相等。

> 那么每个计算单元具体是如何知晓自己的位置的呢？
> 譬如调用<<<64,128>>>的核函数内部，一个thread的blockIdx为1、threadIdx为1，那么它就是全局顺序idx129的thread即i*128+1，假设这个函数功能是向量内元素自增1，并且向量元素和网格内thread相同，那么它就应该处理input[129]++的任务。

那么自然会产生一个疑问：为什么不直接分配一个Block或一个Grid参数，就能解决所有问题呢？

- Block是编程中部分资源分配的单元，并且适合作为数据分片的单位。
- Grid能够在小数据批量中用来表达数据分片后数据形状，即究竟是分为了几片。
- 在模型结构上，Grid、Block还能变相的与计算类型、多头计算、多batch计算对应，具备相当的灵活性。

### 进阶

实际上，了解了上述基本知识后，只要参照对应的数据分配、搬运、计算API，就能够撰写出能跑的并行计算核函数了（譬如简单的加法、规约等计算，只需要通过计算每个thread负责计算的对应部分即可）

想要进一步提高（更好的规约、矩阵乘、Attention算子），就需要了解更细致的硬件结构、Share内存、寄存器、不同存储位置的访问特征、block对应的硬件、thread对应的硬件乃至thread的实际组织形式warp等。因此不宜在软件层面做过多介绍。

## GPU底层架构

想以CUDA**实现**并行计算是很简单的：只需要了解CUDA编程模型就能做到，但是简单的**实现**无法满足我们的初衷：即计算性能的提升。或者应该说，简单的实现与最优的实现有一段不可忽视的距离，这段距离需要我们了解GPU的底层架构：就像了解CPU那样。

### SP

我们一般对GPU最朴素的想象，就是大量简单计算单元的集合，实际上也确实是这样的。SP就是GPU中最小的执行单元，它是计算的实际承担者。虽然现如今的Nvidia计算卡底层架构增加了针对矩阵计算和超越函数计算的专用硬件，但不妨碍SP的重要性：没有SP，代码将失去底层的执行者，就像CPU失去ALU与CU一样。
```c++
int a = b + 1;
```
在GPU上运行时，实际上就是交给众多的SP，分别处理各自的变量a实现的。

软件上的thread落到硬件执行上，就是由SP执行的。

### warp

SP实际上并不是完全一一独立的，而是多个SP组成一个warp（一般是32个为一个warp）。或者更准确的说，应该看作SP是warp的内部组成元素。一个warp内的SP是会相互影响的，假若warp内不同SP执行不同代码将会导致性能下降。

一个warp内只能在一个时刻运行一个指令，也就是说假若一个分支使得warp内部的一部分SP执行A指令，一部分SP执行B指令，那么即使A指令和B指令都只需要1周期，但实际上warp内部仍然需要两个周期来执行完这两种分支指令。

这种现象叫做**warp divergence（分支发散）**：warp内的线程因为条件判断走进了不同的控制流路径，而硬件在一个时刻只能发射一条指令，无法让不同路径的指令同时执行，只能把各条路径串行地走一遍。在执行某条路径时，处于其他路径上的线程会被"遮蔽"（inactive），既不参与计算也不写回结果，只有活跃线程的结果才会被保留。因此发散越严重，warp中平均每条指令真正被利用的线程数就越少，有效吞吐也就越低————这也是为什么在CUDA中要尽量避免线程级别的细碎分支，尤其是在warp内部会分叉的分支。

### SM

虽然名字有点奇怪，但是让人失望的是，SM实际上只是SP和warp上层的管理者，其负责指挥数据搬运、寄存器分配、共享内存与全局内存访问等工作。SM也是实际上决定了硬件分配组合的设计，即一个SM内有多少个warp、多少个指令发射器、最多装载几个Block、多少个专用计算单元等等。

每个block实际上都会分配到一个SM，而无法拆分到多个SM，这在Block大小设计上具有指导作用：譬如每个SM装载64个warp、共32个SM的卡上，假若运行block大小是896（32*28），那么grid大小就只能容纳32个block，每个SM将会有4个空闲warp，因此不如直接设置为1024让所有硬件跑满或者设置更小的block让一个SM内部能容纳更多block。

### Share Mem

共享内存是具备更快访问速度、更小访问延迟的内存，相比显存更适合存储频繁访问的数据，可以在代码中通过"\_\_shared\_\_"声明一个变量或数组分配到共享内存上。

共享内存也是由SM享有的，也意味着在软件层面上，共享内存是block享有的，无法跨block访问。同时共享内存上的数组分配可以是动态的，通过核函数调用时额外的参数来指定其大小。

```c++
__global__ void func(){
	__shared__ int mem[];
	return;
}
int main(){
	func<<<1,64,32>>>();//32个元素的数组分配
	cudaDeviceSynchronize();
	return 0;
}
```

### 内存层级

cudaMemcpy执行的，是将数据从CPU内存搬运到GPU显存（HBM）上的数据搬运，因此访问速度慢，容量大。

GPU访问HBM时存在L2缓存，这一设计使得GPU具备与CPU类似的局部性，相邻warp短时间同时访问向量数据能够提升访问效率，可以视为是HBM的访问特征。

访问shared mem时初衷是让一个warp访问内存时就像计算时一样具备并行性，它的实际访问性质可以看作是一个“多个低位交叉的SRAM”：共享内存被划分成若干个bank（通常是32个），每个bank在一个时钟周期内只能服务一次访问请求。

需要补充的是，bank并不是所有架构上都完全一样：不同计算能力的GPU在bank配置上可能有所差异，例如bank的位宽可以被配置为4字节或8字节（8字节模式下两个相邻的4字节字会落在同一个bank里）。

1. 32个thread分别访问32个连续内存时(sMem[idx]) ，每个thread分别对应一个“SRAM”，能够顺利并行执行。
2. 假若32个thread访问方式使得部分SRAM被多个thread访问(sMem[2*idx])，就会产生串行访问，这一问题称为**Bank Conflict（存储体冲突）**。

冲突的判定条件更精确一些是：只有在**同一个warp**内的多个线程，于**同一条shared memory访问指令**中访问了**同一个bank的不同地址**时，才会发生bank conflict。反过来说，如果多个线程访问的是**同一个地址**，硬件会做**广播（broadcast）**，一个周期就把数据同时送达所有请求它的线程，因此不会产生冲突。

一般解决方法有两个方向。

#### 数据填充

假若设置int M\[32\]\[32\]为共享内存，一个wrap内访问方式为:

```c++
for(int i=0;i<32;i++)sum+=M[idx][i];
```

那么毫不意外的会发生Bank Conflict：每个thread的第i个元素都分配在第i个“SRAM”中，需要32个访存周期才能完成一次循环。

可以通过将M设为M\[32\]\[33\]来解决此问题：M\[i\]\[j\]将会分配到(i+j)%32个“SRAM”，恰好能错开每行的访问，从而通过牺牲空间解决了冲突访问问题。

#### 访问变换

解决bank conflict的方法无非是将每一个时刻不同threads的访问分配到32个不同的“SRAM”上，避开相同"SRAM“上的同时访问。

譬如

```c++
for(int i=0;i<32;i++)sum+=M[idx][(i+idx)%32];
```

更广泛的变换方式是

```c++
for(int i=0;i<32;i++)sum+=M[idx][i^idx];
```

我们不妨讨论一下变换方式的性质：

这种访问变换具备以下性质
$$
f(i,j)=g_i(j)\\
g_i是双射\\
g_{i_1}(j)\%32==g_{i_2}(j)\%32当且仅当i_1==i_2
$$

其中第三条性质值得多说一句：`g_i(j) % 32` 取的就是bank索引，所以这个条件要求的是——**对于同一列 `j`，不同行 `i` 的访问会落在不同的bank上。**

最经典的变换方式就是异或（XOR），它显然具备上述三个性质。

XOR之所以能成为swizzle中最经典的变换，除了这个单射性质之外，还有一个相当漂亮的原因：**XOR是自逆的**——`a ^ b ^ b == a`，也就是说编码和解码可以用同一个函数完成，不需要额外维护一套逆映射逻辑。放在硬件实现上，这意味着地址重映射只需要一两次异或，代价极低。

当然，swizzle的实现路径不止XOR一种，还存在**chunk-based swizzle**等变体。而在NVIDIA Hopper及之后的架构中，专用计算TMA硬件本身就内置了多种swizzle模式（如 `CU_TENSOR_MAP_SWIZZLE_32/64/128B`），软件只需要在描述符里声明使用哪种模式，地址重映射就由硬件自动完成，这也让swizzle与TMA、`ldmatrix` 等硬件指令的配合变得越来越自然。

#### Padding 与 Swizzle 的取舍

到这里其实有了两条路线，它们的适用场景并不一样：

- **Padding（数据填充）** 实现简单直观，不需要修改访存索引，但会浪费shared memory空间；而且行宽越大，padding带来的空间开销就越不可接受，也可能反过来限制能装载的Block数量。
- **Swizzle（访问变换）** 不浪费空间，代价是每次访存多一点点地址计算（通常就是一两次XOR，开销可以忽略），并且与TMA、`ldmatrix` 等硬件指令的配合更好——TMA甚至可以直接在硬件层面完成重映射，所以在新架构的高性能算子中，swizzle是更主流的选择。

## 参考资料

> - [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)[NVIDIA Turing Architecture Whitepaper (2018)](https://www.nvidia.com/en-us/geforce/news/geforce-rtx-20-series-turing-architecture-whitepaper/)
> - [NVIDIA Tesla P100 (Pascal GP100) Whitepaper](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf)
> - [Shuang Gao, Gregory D. Peterson. "Optimizing CUDA Shared Memory Usage." SC15](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post221s2-file3.pdf)
> - [Padding Free Bank Conflict Resolution for CUDA-Based Matrix Transpose Algorithm. SNPD 2014](https://ieeexplore.ieee.org/abstract/document/6888709)
> - [Peyman Afshani, Nodari Sitchinava. "Eliminating Bank Conflicts in GPU Mergesort." SPAA 2025](https://dl.acm.org/doi/10.1145/3694906.3743337)
> - [MLC.AI "Data Layout and Its Notation"](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_data_layout/index.html)
> - [LeetCUDA swizzle位级推导笔记](https://github.com/xlite-dev/LeetCUDA)




# PicoSpec: A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference


# Abstract

Recent advancements and widespread adoption of Large Language Models (LLMs) in both industry and academia have catalyzed significant demand for LLM serving. However, traditional cloud services incur high costs, while on-device inference alone faces challenges due to limited resources. Edge-cloud collaboration emerges as a key research direction to combine the strengths of both paradigms, yet efficiently utilizing limited network bandwidth while fully leveraging and balancing the computational capabilities of edge devices and the cloud remains an open problem. To address these challenges, we propose Pipelined Collaborative Speculative Decoding Framework (PicoSpec), a novel, general-purpose, and training-free speculative decoding framework for LLM edge-cloud collaborative inference. We design an asynchronous pipeline that resolves the mutual waiting problem inherent in vanilla speculative decoding within edge collaboration scenarios, which concurrently executes a Small Language Model (SLM) on the edge device and a LLM in the cloud. Meanwhile, to mitigate the significant communication latency caused by transmitting vocabulary distributions, we introduce separate rejection sampling with sparse compression, which completes the rejection sampling with only a one-time cost of transmitting the compressed vocabulary. Experimental results demonstrate that our solution outperforms base-Draft Model line and existing methods, achieving up to 2.9× speedup.



# 3 Framework Design

# 3.1 Overall Framework

The PicoSpec framework aims to achieve efficient and fully parallelized collaborative speculative decoding in edge-cloud scenario, while simultaneously addressing the challenges posed by distributed communication.

The system overview of PicoSpec is illustrated in Figure 2.

![](images/ecaf4a9eab8339eb9c0ebce950faf1e67f7fa152e974e21e5e0aefb746eea52c.jpg)  
Figure 2: System Overview of PicoSpec Framework.

Our framework comprises two main components: the edge-side and the cloud-side. On the edge side, it includes four modules: Parallel Drafter, Rejection Sampler, Speculative KV Cache, and Zero-Copy Communicator. On the cloud side, it consists of four modules: Verifier, Request Handler, KV Cache, and Zero-Copy Communicator.

After the user prepares the prompt and sampling parameters, both the edge and cloud simultaneously enter the prefill stage to initialize the request’s KV Cache. Subsequently, the user request proceeds to the collaborative speculative decoding stage. The edge-side Parallel Drafter module first executes drafting in parallel. Data is then transferred efficiently via the Zero-Copy Communicator module. The cloud-side Request Handler handles different types of verification requests. Finally, the Rejection Sampler module on the edge side completes the rejection sampling process to determine the final output. PicoSpec fully leverages KV cache to accelerate model execution. We also implemented an efficient rollback KV cache manager within the KV Cache modules on both the edge and cloud sides.

Decoupled design lays the foundation for parallelism, but there are still two main problems. First, how to fully use parallelism, the edge needs to ”speculate ahead” with draft generation while waiting for cloud verification, which requires precise scheduling to avoid pipeline bubbles and ensure execution continuity. To address this, we propose Parallel Drafting and Fast Verification in Section 3.2.

Second, the system faces a communication-overhead trap. In a WAN environment, even a small increase in message frequency or data size can quickly saturate the limited bandwidth and negate any gains from speculation. Efficiently pruning data and fusing requests without losing model accuracy is a delicate balancing act. We address these critical efficiency issues through the optimized sampling and com-Drafter Executormunication techniques described in Section 3.3.

The lifecycle of PicoSpec, shown as Figure 3, moves from a synchronous prefill to a continuous execution loop. After the initial prefill aligns the context between the edge and cloud, the cloud provides a seed token to trigger the speculation process. In the core Asynchronous Speculative Loop, the edge follows Parallel Drafting, drafting the next batch while the cloud verifies the previous one. To further cut down on waiting time, we use Fast Verification step that allows the cloud to start its work before the full draft even arrives. Meanwhile, we introduced Overlapped Communication, the data transmission time between edge-clouds is also successfully masked. If the cloud rejects a token, it immediately sends an interrupt signal to stop the edge’s now-invalid drafting tasks. The system then performs a rollback and correction phase to restore state consistency before resuming the loop.

# 3.2 Edge-Cloud Collaborative Speculative Decoding

PicoSpec decouple the local drafting logic from the remote verification process through asynchronous pipeline. In order to achieve better performance, achieving true edge collaborative speculative decoding. It is necessary to solve the data dependency problem at the edge and the cloud side.

In traditional synchronous stop-and-wait mode, the end-toend latency for a single inference step, denoted as $L _ { s y n c } .$ is the linear sum of each phase’s duration:

$$
L _ {\text { sync }} = T _ {\text { draft }} + T _ {\text { RTT }} + T _ {\text { verify }} \tag {1}
$$

Here, $T _ { d r a f t }$ represents the total time for the edge to generate γ draft tokens, $T _ { R T T }$ is the network round-trip time, and $T _ { v e r i f y }$ is the time required for parallel verification on the cloud. Due to strict data dependencies in this mode, the edge must keep idle during the $\bar { T _ { R T T } } + T _ { v e r i f y }$ interval while waiting for the cloud to respond.

As shown in Figure 4, PicoSpec solves the data dependency problem of edge side by introducing Parallel Drafting. In the main generation loop, once the model execution thread completes drafting the current batch $S _ { i }$ and submits a verification request, control is immediately returned to the system via a pre-draft. Instead of waiting for remote feedback, the main thread assumes $S _ { i }$ is fully accepted. It uses the tail state of $S _ { i }$ as the logical context to immediately start generating the next batch, $\bar { S _ { i + 1 } }$ . Ideally, when $T _ { d r a f t } \geq T _ { R T T } + T _ { v e r i f y } ,$ the verification of step i and the generation of step i + 1 overlap completely on the timeline. Under these conditions, the amortized latency $L _ { a s y n c }$ becomes:

$$
L _ {a s y n c} = \max (T _ {d r a f t}, T _ {R T T} + T _ {v e r i f y}) \tag {2}
$$

This design transforms the serial dependency chain of tra-Zero-Copy Communicator Zero-Copy Communicatorditional cooperation speculative decoding into a parallel execution flow. As a result, system throughput is no longer bound by RTT and is instead limited only by the continuous productivity of the edge’s draft generation.

To solve the problem of data dependence on the cloud side, PicoSpec propose Fast Verification mechanism. This mechanism aims for a secondary overlap of computation and communication at a sub-batch level. While standard asynchronous modes wait for the complete draft sequence $S _ { i }$ to arrive at the cloud before starting verification, Fast Verification allows the PreVerify control signal as soon as partial tokens in $S _ { i + 1 }$ are determined. Figure 5 illustrates the difference before and after the introduction of Fast Verification.

If we define the pre-verify lead time as $T _ { p r e } ,$ , the effective start time of cloud verification, $T _ { s t a r t } .$ , is moved forward.

![](images/6b1f19ca3b1476921955fa7bb067e7297bc03142f9556b924da8b60f77ad3895.jpg)  
Figure 3: Asynchronous Pipeline of PicoSpec. ❶ Parallel Drafting: After the first draft, we perform multiple Pre-Draft steps consecutively. The i-th Pre-Draft on the edge is executed simultaneously with the (i-1)-th Verification in the cloud. ❷ Fast Verification: When a verification failure occurs, we immediately trigger a Pre-Verifiy step, and the subsequent Verify step quickly returns the verification result of the first token. ❸ Overlapped Communication: The sending/receiving of data stream on the edge overlaps with Draft step.

![](images/fb01fa1ee90c634c65f2f416e449c9e1a5a1f41cbe33534a54f8daff8166b171.jpg)  
Figure 4: Parallel Drafting in asynchronous pipeline.

![](images/e440de6d569dfdf960cb02b382147408c91e7844ea02d70d5291ce56996fe5be.jpg)  
Figure 5: Fast Verification in asynchronous pipeline.

Consequently, the bubble time $T _ { b u b b l e }$ in the pipeline is further reduced to:

$$
T _ {b u b b l e} = \max (0, T _ {R T T} + T _ {v e r i f y} - T _ {d r a f t} - T _ {p r e}) \tag {3}
$$

By using the powerful parallel forward-computation capabilities of the cloud server, this strategy allows the verifier to begin preparation based on a known prefix even before the V0 1Pre Vedge has finished its entire speculative task. Even in complex scenarios with network jitter or fluctuating acceptance rates, the Fast Verification mechanism ensures a smooth inference pipeline through tighter time-slot scheduling. This computa-

tional head-start logic provides PicoSpec with strong ”latency immunity” against long-distance delays.

# 3.3 Efficient Separate Rejection Sampling Algorithm

To avoid the communication-overhead trap, PicoSpec uses a specialized algorithm to minimize the amount and the frequency of data exchanged between the edge and the cloud.As shown in Figure 6.

In standard distributed setups, transmitting full vocabulary distributions is often too expensive for WAN bandwidth. To solve this by introducing a decoupled resampling method. Unlike traditional distributed speculative decoding that the cloud handles both verification and correction, during each step, instead of full vocabulary distributions, the edge only records and transmits the specific probabilities $( q _ { 1 } \ldots q _ { \gamma } )$ corresponding to these candidates. This reduces the uplink payload to less than 50 bytes per round, significantly lower than the tens of kilobytes required in conventional designs. The cloud LLM performs parallel verification to determine the acceptance of each candidate. If a token is rejected at position j, the cloud identifies the divergence and prepares the target distribution $P _ { j } ( x )$ . Downlink transmission occurs only upon rejection. The edge then autonomously performs local resampling using the logic norm(max $( \bar { 0 , + } \bar { P _ { j } } ( x ) - Q _ { j } ( x ) ) ;$ ) to produce the corrected token.

After verification, return the edge model’s vocabulary size V , where typically 128, 256 for Llama-3.1 70B and up to 151, 936 for Qwen3. Transmitting such huge data creates significant bandwidth pressure. To further compress the downlink payload, PicoSpec uses a sparse compression strategy on probability data. The cloud extracts only the K components saved time with the highest probabilities and their corresponding indices:

![](images/44aab15663a4fc52720c654b3bc22e69aacafe59d21c96053dbb16ad00b28aef.jpg)  
Figure 6: (Left) Vanilla rejection sampling & (Right) Ours

$$
P _ {s e n t} = \left\{\left(i d _ {j}, p r o b _ {j}\right) \mid j \in \{1, \dots , K \}, \right.
$$

$$
\left. \operatorname{prob} _ {j} \in \operatorname{TopK} \left(P _ {\text {auth}}\right) \right\} \tag {4}
$$

Since LLM probability distributions are typically very long-tailed, Top-K pruning can reduce the transmission load by two orders of magnitude while keeping most of the ”probability mass”. By extracting only the Top-K probability components and their corresponding indices, PicoSpec reduces the downlink payload from $O ( \bar { V } )$ to $O ( K )$ . For instance, with $V = 1 2 8$ , 256 and $K = { \dot { 1 } } 0$ , the transmission load per rejected token drops from approximately 500 KB to less than 100 bytes—a reduction of over three orders of magnitude. Combined with our zero-copy serialization protocol, the edge can reconstruct tensor views in microseconds.

Finally, to handle the network jitter and changing edge loads, we include a latency-aware truncation mechanism. In the real world, if generating a draft takes too long, the cloud server will sit idle, wasting its massive computing power. PicoSpec monitors the drafting time in real-time. If the edge is struggling to keep up, the system automatically stops the current speculation, cuts the segment short, and sends it immediately. This dynamic adjustment keeps the pipeline flowing smoothly even when the environment is unstable, masking the latency.

# 3.4 Framework Performance Analysis

To explore the performance boundaries of PicoSpec, we conduct a formal analysis of end-to-end throughput using a probabilistic model.

corrected tokenDefine the following parameters: γ as the speculative step size; α as the token acceptance rate (the independent probability that a draft token is accepted by the target model); $T _ { d r a f t }$ as the total time for the edge to generate γ draft tokens; $T _ { v e r i f y }$ as the time for the cloud to verify γ tokens; and $T _ { R T T }$ as the network round-trip time, including serialization and deserialization overhead.

In a single speculative cycle, the number of successfully generated tokens, L, follows a truncated geometric distribution. The expected generation length EL can be expressed as:

$$
E L = \sum_ {k = 1} ^ {\gamma} k \cdot P (L = k) = \frac {1 - \alpha^ {\gamma}}{1 - \alpha} \tag {5}
$$

As $\alpha  1 , E L  \gamma ; \mathrm { a s } \alpha  0 , E L  1$ .

In a traditional synchronous stop-and-wait mode, the edge must pause after generating drafts to wait for cloud verification. The end-to-end latency for a single cycle, $L _ { s y n c } .$ , is the linear sum of the time required for each phase:

$$
L _ {s y n c} = T _ {d r a f t} + T _ {R T T} + T _ {v e r i f y} \tag {6}
$$

In this mode, $T _ { R T T }$ and $T _ { v e r i f y }$ are unavoidable costs on the critical path, making the overall speed significantly lower than purely local inference. The theoretical throughput $R _ { s y n c }$ is defined as:

$$
R _ {s y n c} = \frac {E L}{L _ {s y n c}} = \frac {1 - \alpha^ {\gamma}}{(1 - \alpha) (T _ {d r a f t} + T _ {R T T} + T _ {v e r i f y})} \tag {7}
$$

This formula highlights that network latency $T _ { R T T }$ severely limits the throughput ceiling in synchronous systems.

PicoSpec breaks these serial dependencies through its asynchronous scheduling. By Parallel Drafting, the verification of step i and the generation of step i + 1 occur in parallel. However, the actual efficiency of the pipeline depends on the accuracy of the speculation:

• Case 1: Full Hit $( L = \gamma )$ . The pre-computation for step $i + 1$ is valid, keeping the pipeline full and effectively hiding the network latency. This occurs with probability $\bar { P _ { h i t } } \bar { = } \alpha ^ { \gamma }$ Edge Draft Model. The amortized time is $T _ { h i t } =$ max $( T _ { d r a f t } , T _ { R T T } + T _ { v e r i f y } )$ .   
• Case 2: Miss $( L < \gamma )$ . Step i + 1 is generated based on an incorrect context and must be discarded (pipeline flush), resulting in a bubble. The execution effectively reverts to a serial pattern with probability $P _ { m i s s } = 1 -$ $\alpha ^ { \gamma }$ . The time cost is $T _ { m i s s } = \dot { T _ { s y n c } } = \dot { T _ { d r a f t } } + T _ { R T T } +$ Tverify. $T _ { v e r i f y } .$

Accordingly, the expected amortized time per step for PicoSpec, $E T _ { a s y n c } ^ { - } ,$ is calculated as:

$$
E [ T _ {a s y n c} ] = P _ {h i t} \cdot T _ {h i t} + (1 - P _ {h i t}) \cdot T _ {m i s s} \tag {8}
$$

$$
\begin{array}{l} E \left[ T _ {\text { async }} \right] = \alpha^ {\gamma} \cdot \max \left(T _ {\text { draft }}, T _ {\text { RTT }} + T _ {\text { verify }}\right) \\ + (1 - \alpha^ {\gamma}) (T _ {\text { draft }} + T _ {\text { RTT }} + T _ {\text { verify }}) \tag {9} \\ \end{array}
$$

The theoretical throughput of PicoSpec is then $R _ { a s y n c } =$ EL/ETasync. $E L / E T _ { a s y n c } .$

Define the theoretical speedup as $S = R _ { a s u n c } / R _ { s u n c } 1 3$ . In a compute-bound scenario $( \bar { T } _ { d r a f t } \geq T _ { R T T } + \bar { T _ { v e r i f y } } )$ with high model alignment $( \alpha \approx 1 )$ , the performance gain reaches its upper bound:

$$
\lim _ {\alpha \rightarrow 1} S = \frac {T _ {\text { draft }} + T _ {R T T} + T _ {\text { verify }}}{T _ {\text { draft }}} = 1 + \frac {T _ {R T T} + T _ {\text { verify }}}{T _ {\text { draft }}} \tag {10}
$$

This theoretical framework leads to two important conclusions:

• Latency Immunity: Provided α is sufficiently high, PicoSpec can effectively remove $T _ { R T T }$ from the bottleneck, allowing collaborative inference speeds to approach the physical limit of local-only draft inference.   
• Robustness Lower Bound: Even in the worst-case scenario $( \alpha  0 )$ , PicoSpec adaptively degrades to the performance of the synchronous baseline (S → 1) rather than collapsing due to communication overhead.

# 4 Evaluation


# 4.3 Ablation Experiment

To investigate the contribution of each core component in PicoSpec, we conduct an extensive ablation study by isolating three key mechanisms: asynchronous pipelining (Paradraft), Fast Verification(Fast-verify), and separate rejection sampling (Split-rej). The results, detailed in Table2 & 3, demonstrate that the synergy of these components is essential for maintaining high performance in high-latency environments.

The removal of the asynchronous pipeline (w/o Paradraft) results in the most significant performance degradation. Without this mechanism, the system reverts to a traditional serial stop-and-wait paradigm where the edge-side NVIDIA Jetson AGX must remain idle during the entire $T _ { R T T } + T _ { v e r i f y }$ interval. For the Llama-gsm8k setting, the throughput drops from 17.22 tokens/s to 12.51 tokens/s. This confirms that decoupling computation from communication is the primary driver for masking network latency.

The Separate Rejection Sampling algorithm is important for minimizing communication overhead. When this component is disabled (w/o Split-rej), the system is forced to transmit huge probability distributions instead of specific probabilities and compressed results. This leads to a massive surge in $T _ { v e r i f y }$ due to increased bandwidth pressure and CPU deserialization costs. Specifically, in the Llama-gsm8k scenario, $T _ { v e r i f y }$ jumps from 166.46ms to 283.79ms, highlighting the necessity of communication-efficient sampling for distributed speculative decoding.

The Fast Verification strategy aims to further eliminate residual bubble in the pipeline by initiating cloud preparation before the entire draft batch is finalized. Removing this feature (w/o Fast-verify) consistently increases the Time Per Output Token (TPOT). For instance, in the QWEN-gsm8k case, the throughput decreases from 20.19 to 17.87 tokens/s. This illustrates that Fast Verification provides an essential computational head-start, ensuring a smooth inference flow even when network jitter occur or low acceptance rate.

Table 2: Ablation results for the Qwen model pair (0.6B and 32B). 

<table><tr><td>Configuration</td><td>Thr. ↑ (tokens/s)</td><td>TTFT ↓ (ms)</td><td>TPOT ↓ (ms)</td><td> $T_{verify}$  ↓ (ms)</td><td> $T_{draft}$  ↓ (ms)</td></tr><tr><td colspan="6">Dataset: gsm8k</td></tr><tr><td>w/o Para-draft</td><td>11.92</td><td>148.96</td><td>251.76</td><td>119.81</td><td>123.91</td></tr><tr><td>w/o Fast-verify</td><td>17.87</td><td>150.32</td><td>166.56</td><td>106.56</td><td>97.30</td></tr><tr><td>w/o Split-rej</td><td>12.62</td><td>154.68</td><td>233.30</td><td>166.98</td><td>100.96</td></tr><tr><td>Ours (Full)</td><td>20.19</td><td>143.99</td><td>148.84</td><td>87.35</td><td>97.46</td></tr><tr><td colspan="6">Dataset: humaneval</td></tr><tr><td>w/o Parallel</td><td>10.25</td><td>177.90</td><td>252.92</td><td>119.03</td><td>124.08</td></tr><tr><td>w/o Fast-verify</td><td>13.86</td><td>164.84</td><td>187.07</td><td>109.14</td><td>97.16</td></tr><tr><td>w/o Split-rej</td><td>10.23</td><td>164.82</td><td>251.73</td><td>168.27</td><td>99.98</td></tr><tr><td>Ours (Full)</td><td>16.04</td><td>155.11</td><td>160.88</td><td>78.04</td><td>97.17</td></tr></table>

Table 3: Ablation results for the Llama model pair (1B and 70B). 

<table><tr><td>Configuration</td><td>Thr. ↑ (tokens/s)</td><td>TTFT ↓ (ms)</td><td>TPOT ↓ (ms)</td><td> $T_{verify}$  ↓ (ms)</td><td> $T_{draft}$  ↓ (ms)</td></tr><tr><td colspan="6">Dataset: gsm8k</td></tr><tr><td>w/o Parallel</td><td>12.51</td><td>305.46</td><td>390.35</td><td>194.68</td><td>190.56</td></tr><tr><td>w/o Fast-verify</td><td>16.64</td><td>297.40</td><td>287.74</td><td>189.32</td><td>170.93</td></tr><tr><td>w/o Split-rej</td><td>12.12</td><td>303.17</td><td>389.76</td><td>283.79</td><td>173.69</td></tr><tr><td>Ours (Full)</td><td>17.22</td><td>295.58</td><td>272.89</td><td>166.46</td><td>171.05</td></tr><tr><td colspan="6">Dataset: humaneval</td></tr><tr><td>w/o Parallel</td><td>13.60</td><td>342.68</td><td>380.89</td><td>186.60</td><td>189.68</td></tr><tr><td>w/o Fast-verify</td><td>19.67</td><td>346.16</td><td>263.94</td><td>182.32</td><td>171.64</td></tr><tr><td>w/o Split-rej</td><td>14.29</td><td>344.35</td><td>366.93</td><td>281.84</td><td>173.60</td></tr><tr><td>Ours (Full)</td><td>19.88</td><td>369.02</td><td>272.93</td><td>182.96</td><td>172.41</td></tr></table>

# 4.4 Parameter Sensitivity

This section investigates how the draft generation length (n) affects PicoSpec’s end-to-end performance, using the QWEN-gsm8k setting as a representative case study. As shown in Table 4, the choice of n involves a critical trade-off between speculative gains and local computational overhead. We observe that while the average acceptance length grows steadily from 2.03 to 3.10 as n increases, the system throughput remains remarkably stable within the n = 3 to n = 5 range, staying above 18 tokens/s. This show that PicoSpec is robust and does not require extremely precise parameter tuning to achieve significant speedups.

The throughput reaches its peak of 20.19 tokens/s at n = 4. This represents the draft generation time $( T _ { d r a f t }$ ≈ 97ms) effectively overlaps with the sum of network RTT and cloud verification time, successfully masking the communication latency. However, as n grows beyond this point, the performance begins to decline. This is because the local drafting cost starts to escalate—rising from 82.11ms at $n = 3$ to 145.40ms at n = 6—which eventually outweighs the benefits of a higher acceptance rate. Consequently, the Time Per Output Token (TPOT) increases significantly, and the system moves away from its ideal non-blocking state. Based on these observations, we select n = 4 as the default configuration for our framework.

Table 4: Performance comparison under different draft generation lengths. 

<table><tr><td>Draft Len (n)</td><td>Thr. ↑ (tokens/s)</td><td>Acc. Len ↑</td><td>TTFT ↓ (ms)</td><td>TPOT ↓ (ms)</td><td> $T_{verify}$  ↓ (ms)</td><td> $T_{draft}$  ↓ (ms)</td></tr><tr><td>3</td><td>19.06</td><td>2.03</td><td>146.55</td><td>129.15</td><td>82.11</td><td>73.41</td></tr><tr><td>4</td><td>20.19</td><td>2.50</td><td>143.99</td><td>148.84</td><td>87.35</td><td>97.46</td></tr><tr><td>5</td><td>18.12</td><td>2.84</td><td>143.39</td><td>189.17</td><td>88.76</td><td>121.40</td></tr><tr><td>6</td><td>17.12</td><td>3.10</td><td>143.26</td><td>219.28</td><td>88.07</td><td>145.40</td></tr></table>

# 5 Conclusion

This paper present PicoSpec, a plug-and-play framework designed to overcome the communication-computation mismatch in Edge-Cloud speculative decoding. By decoupling edge-side drafting from cloud-side verification through an asynchronous pipeline, PicoSpec effectively masks roundtrip times without requiring model modifications or retraining. Experiments conducted using an NVIDIA Jetson AGX and an A100 cluster demonstrate that our framework achieves significant ”latency immunity”. For large models like Llama-70B, the method can run up to 2.9 times faster than cloudonly autoregressive models. While PicoSpec have strong performance, its speedup remains naturally bounded by the alignment between the draft models and target models. But, the system can ensures robustness by degrading to a synchronous baseline in the worst-case.

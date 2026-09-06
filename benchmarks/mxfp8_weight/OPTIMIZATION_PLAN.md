# SM90 MXFP8-weight 四算子优化计划

## 起点

优化 goal 已建立。`operators/full_v1/cp4.json` 和 `cp8.json` 已全部完成并通过
审计：384 settings、2,304 records、无缺项。初版库另存为同目录
`libfuse_mxfp8_baseline.so`，SHA256 为
`8219d20ba747f55033e47e81528ef39e26dc7e92a5538874e3d1b584d0ec6695`。
封存本分支的初版生产算子、最优外部基线、经典 cuBLAS 与配置。优化前后必须
能回到各自的源码、库哈希和独立正式采样，不能覆盖初版结果。

初版长序列完整边界等权几何平均 1.132×，四算子等权 1.140×；全量分别
1.024× / 1.026×，所有阶段目标尚未达到。逐组数据见 comparison 下的
`optimization_summary.md`。CP4 Graph 普通反向的长序列 B 段均值为
QKV 1.225× / OProj 1.265×，W 段为 0.964× / 0.940×；这是优先诊断
FP32 W GEMM 的实测依据，还不是 tile 或流水修改已有效的结论。

两个 pilot 的逐样本文件已被正式覆盖后清理，共 468,633 bytes；没有删除初版
全量向量、验证记录或外部基线。后续 profile 只保留每 CTA 汇总和 Perfetto，
不保留另一个重复 CTA 原始导出。

## 验收

根据用户最新指示，先完成一组新版四算子全量，展示效果后再针对薄弱项优化。
当前目标收敛为 1.2× 第一验收、1.3× 冲刺；早期 1.4×/1.5× 仅保留历史与
假设对照，不再作为硬指标。goal 初始文本中的四阶段安排由此更新。

- 重点是展平 token 数 128K/256K/512K，16K 检查性能过渡；
- 1K/4K 保留正确性、全量覆盖和回退报告，不为短点牺牲长序列；
- 长序列几何平均的阶段目标：1.2× 第一验收、1.3× 冲刺；
- 分母是同 shape、CP、启动方式和反向模式下的最优 TEUB/cuBLASLt 完整边界；
- 四算子、CP4/CP8、Eager/Graph、普通/分离反向分别报告，同时公布全量结果；
- 经典 cuBLAS 是计算吞吐参照，不是同边界的外部基线，也不作为不可突破的理论上限。

初版全量的计算参考：假设通信与 DQ 都免费，沿用实测 classic cuBLAS 的
GEMM 速度，长序列对外部基线仅为 1.32859×（边界等权）/1.34941×（四算子
等权）。反向使用实际同流 B→W total，不相加孤立 B/W median。它不是硬件
绝对上限：已有三个长序列 Fuse 边界快于 classic cuBLAS。全量没有独立 DQ
正式分项，不能用少量 marker profile 外推“保留独立 DQ”的严格参考。
新的 `optimization_report.py` 与全量结果一起重算该参考及剩余空间。

## 方法

最新工作方法：一次只分析一个 scope，先按正式全表筛出 badcase，宏 profile
区分 DQ、compute、route 暴露与跨 rank finalize，再解释瓶颈分化，测量 tile/
通信 CTA 的收益与代价，最后修改并验证泛化性能模型。目标是通信跟上计算、
总时延下降，而不是靠降低计算速度把所有点强行变为同类瓶颈。
长序列优先测通信/计算的稳态资源交叉点：通信占用 c 个 SM、计算占用其余
SM，整数 wave 尾部修正后置。按本 scope 优化前的固定几何平均筛选低于平均
的点，不随改善结果移动门槛；逐点争取达到 min(原 scope 均值, 该点可达上限)。
其中上限必须写清是假设参考还是严格界，不能把 classic cuBLAS 速度当成
硬件理论上限，也不能仅由加速比低就认定某个硬件单元没有打满。
两类物理方向是 GEMM→A2A 与 A2A→GEMM，但四条 F/B 算子的路由、权重方向
和计时边界仍分别验收，不用同方向的一条结果代替另一条。

上一阶段 scope 由用户指定为 CP4 的 QKVProj Forward→A2A，T=128K/256K/512K。
Q2/Q3/Q5/Q7/Q8 存在 Eager 或 Graph 小于 1.1× 的点；其他长点作为对照。
最新筛选门槛改为该 scope 正式 Eager/Graph 48 个边界的原 GM 1.123583×；
共有 27/48 个边界低于这个固定门槛，包含此前漏在 <1.1× 表外的 Q4/256K
Eager。1.1× 保留为前一轮筛选记录，不是新的硬编码优化目标。
该阶段未同时推进 QKV backward、OProj 或 CP8；后续顺序切换见本节最新记录。
24 个自动配置宏 profile 已完成，53 项来源哈希和所有 rank 的配置与正式表
核对通过；结果保存在 `profile/qkv_forward_cp4_long_v1`。Q2/Q5/Q7 的
512K route 未覆盖比例分别约 21%/29%/21%；Q3/Q8 不足 0.2%。该指标
按同 rank 的 route 包络减去 overlap 计算，不是裸通信时间。
另外三个 512K 代表点的 Perfetto 重复采样确认：Q3/Q8 的同 rank 计算结束后
路由尾部只有几十微秒，Q5 约 3.6–3.8ms。正式胜负仍以 profiling OFF 为准。

固定 M128N256K64/C2、c=4/8/12/16/24 的 CP4 全八种几何 128K/512K
角色曲线已完成（80 配置/320 rank），保留 256K 作通信模型长度验证。
校准摘要只保留紧凑 rank 统计；不得把 profile 最优点直接写成 shape dispatch。
关闭 profiling 的 16 点三轮交错 10+50 CTA A/B 已完成并通过审计。
对计算角色主导的 Q3/Q8 另外测试合法最小 c2：C2 下计算 SM 从 128 增至
130，纯资源缩放的理想收益仅约 1.016×，通信槽却减半，因此需要实测，不能
预先认定有效。该对照包含三个长序列，但不用于拟合通信模型的 256K 留出集。
这六点 c2 对照也已完成：两轮数据共 228 个正式 arm、11,400 个逐样本
rank-max，全部正确性、来源与实际配置通过。512K 相对原 auto 的三轮 GM：
Q2/c8 1.1145×、Q4/c12 1.2470×、Q5/c12 1.1826×、Q7/c8 1.1284×。
Q3/Q8 增 c 均退化，减至 c2 只有约 0–1.2% 且部分不稳定，不能仅靠增加两个
计算 SM 补齐其主要差距。128K 的 auto 漂移另报，不把候选表当通用 dispatch。
完整诊断与两套正式逐点表在 `profile/qkv_forward_cp4_long_v1/diagnosis.md`。

同一 GEMM/scheduler 的 producer 独立服务已完成：signaling/null-ready
只改变同一 kernel 的 ready 指针，分别测 subgrid/fullgrid；另测匹配 fused
cluster/SMEM 的原 route。用于区分逐 tile store-drain/publication、资源缩放
和并发竞争，不将独立服务直接相减称作裸通信或硬件 stall。新测量复用现有
`calibrate_operators.py`，不添加一次性的 runner 或批量 CTA trace。
5 个短点及 Q3/Q5/Q8 的 128K/512K 共 30 个长点通过正确性与正式 10+50；
它们是独立原语参考，不是配对的 signaling 开销测量。Q8/512K 即使去掉
ready publication、给满计算网格仍约 122ms，不能把这当硬件绝对上限。

另用 Nsight Systems 只读 GPU metrics 采样 Q8/128K 独立 producer，不改
时钟、boost、缓存状态。GPU0 正式 50 次 kernel 内部窗口的 SM Active
96.98%、Tensor Active 94.95%，DRAM 读/写接口活动率 57.32%/1.00%；
活动率不是实测 GB/s，也不是 Tensor Core FLOPs 峰值利用率。内窗覆盖
93.56%，GPC 均频约 1290MHz，时延与均频倒数相关系数 0.998312。
这说明不能把 idle/max clock 当负载时钟，几%的跨轮差也不能直接归因于
优化。保留原时延，正式对照使用相同边界与两种 A/B 顺序，未进行频率归一化。
这份带采样的 trace 只作诊断，不替代 profiling OFF 的正式性能。

稳态角色模型的预声明实验（尚非生产策略）：仅以 80 个固定 tile 宏样本拟合，
不使用融合 winner/外部基线，令 G=2MNK/1e9、tasks=MN/(64×128)，
C=a·G/(132−c)、R=tau·tasks/(12c)，选择偶数 c∈[4,24] 的 min max(C,R)。
冻结 a=171.676018685189 µs·SM/GFLOP、tau=5.913378755546 µs/slot-task；
compute 对数误差最小化，route 按 max(实测 C,R) 拟合被计算遮住的包络。
按 (N,K) 留一几何时选择稳定，但 Q8 compute 误差约 13%、两种长度的通信
服务率有漂移，因此只能作为待验证候选，不能称校准完成。预声明 Q2/Q7 的
交叉候选为 c6；三个长序列的 0/c6 正式 A/B 已完成，256K 没有参与拟合。
Q2/Q7 在 256K、512K 的三轮均改善，128K 有跨轮退化，不能隐去。
该简化模型对 Q5 预测 c8 而已知 c8 仍有尾部，也必须保留为模型不足。
模型保存为 `calibration/qkv_forward_service_model_v1.json`，必须显式加载，
生产默认不变。当前 CP4 长序列 scope 的正反两序完整 A/B 已完成：每序
24 settings × Eager/Graph × 三轮 × 两臂 × 50，合计 28,800 个 rank-max
样本，所有正确性、实际配置和两序源码/DSO/model 一致性通过。
相对原 auto 的 48 边界 GM=1.065537×；改变配置的 24 边界全部改善，
GM=1.134815×；不变的 24 边界 GM=1.000488×，保留其最大约 1.96% 退化。
相对归档外部最佳基线为 1.166200×，不是将原 1.123583× 乘以本轮收益：
本次 auto 比原 Fuse 时延 GM 慢约 2.66%，不能隐藏这个跨批次差异。
原 27 个低于均值的边界有 14 个达到原 GM；还剩 Q3/Q8 全部 12 个边界
和 Q7/128K Graph。这里只是当前 scope，未达到四算子 1.2× 里程碑。
全表见 `policy_ab/qkv_forward_cp4_long_summary/policy_summary.md`。

下一步以当前模型实际配置采完整 24 点宏摘要。新增 profile-only `--warmup 10`
复用捕获后的 graph，默认仍为 2；另有 Graph 前两次 eager 初始化，不能把
一次诊断 capture 称为正式三轮 10+50 的相同运行历史。Q7/256K 两序已观测到
auto/model 各自的时延改变，既不能挑末尾较快样本，也不能在没有该点运行中
时钟的情况下直接归因为频率。先检查当前 c6 的 C/R/overlap/finalize，再决定
是否需要同状态 c4/c6/c8 或硬件计数器对照，不改模型去拟合完整 F winner。
当前模型 24 点宏摘要已完成，96 rank 的实际配置与正式 model arm 一致、
正确性通过。Q2/c6 三长度 route 未覆盖约 13/11/19µs；Q7/c6 约 22/33/15µs，
本次采样未显示其继续受通信尾部限制。Q7/256K 的最大计算角色约 6.92ms，
diagnostic F 约 7.05ms；但其正式两序候选 p50 分别约 8.06–8.19ms 与
7.46–7.71ms，不能拿更快的一次 profile 覆盖正式数值或宣布状态问题消失。
Q3/Q8 仍由计算角色主导。进一步诊断应在更接近正式持续运行的状态下复现，
不因为单次通信已覆盖就追加通信 SM。新摘要仅一份：
`profile/qkv_forward_cp4_model_v1/calibration_summary.json`。

状态诊断继续限定 QKV F：Q2/Q7 的 128K/256K 改为 50 次 profile 预热后，
四点正确性通过；Q7 diagnostic F 仍约 3.51/7.09ms，不能用“只预热两次”
解释它与正式持续采样的差异。当前生产/telemetry kernel 的实测寄存器均
168/thread、动态 SMEM 214016 B、local memory 为 0；两库中的生产 SASS
编码完全相同。没有证据支持“打点版靠降低寄存器/消除 spill 提速”。

另在 profiling OFF 库的真实三轮 10+50 A/B 流程下采 Q7/128K Graph，
显式 `--diagnostic-trace` 将结果标为非正式。GPU0 10kHz 内窗覆盖约
94.6–95.1%；c6 三轮 event rank-max p50 为 3.750/4.148/3.832ms，
对应内窗加权 GPC 约 1449/1299/1450MHz、Tensor Active 86.4/82.4/84.7%。
观察到了状态变化，但只有 GPU0 计数器，且相关性不是因果证明；不能把
所有波动归因于频率，也不做频率归一化或用诊断较快一轮更新正式表。
四卡各 370 次 kernel 与采样序号核对，全部正确性通过。正式汇总器新增
拒绝外部 trace 的检查；默认 runner 执行序列不变，生产策略也不变。
证据集中于 `profile/qkv_forward_cp4_model_state_v1/`，详细解释追加在原
`profile/qkv_forward_cp4_long_v1/diagnosis.md`，不增加逐 shape 测试脚本。

随后补齐 Q3/Q8 计算 tile 的同进程交错对照：复用 `operator_sweep.py`，
固定 c4，短扫既有 N64/128/160/192/256/320（M128/K64），不新增内核。
两个 4K case 的全部 12 个配置通过正确性；三个长序列共 36 个短搜配置、
72 个正式 arm、3,600 rank-max/14,400 raw rank 样本也通过原始向量、
配置、来源和 correctness 审计。N320/N256 三轮 GM 为 Q3
1.02114/1.01927/1.01212×、Q8 0.99104/0.99588/1.01953×；
正式窄 tile 对照只有 0.603–0.661×。这否决了“普遍换窄 tile”或
“普遍换 N320”的优化方案，但不是所有 tile 或硬件的绝对上限。
只有 Graph、单一 A/B 顺序，几%的正收益不写入模型，更不按 Q3/Q8 特判。
源码无生产策略变化，QKV F 已验证的主要收益仍来自通信预算模型。
完整原始值与紧凑逐点表为 `sweep/cp4_qkv_fixed_comm_tiles_v1{,_summary}.json`，
解释和精确表追加到原 diagnosis 文档。

当前顺序切换到 CP4 QKV Backward（A2A→GEMM B，完整目标仍含 W），不同时
调其他 scope。以原完整反向 96 个长边界 GM=1.115615655873× 冻结 34 个
低于均值的边界：Q2/Q3/Q7/Q8 分别 9/12/1/12 个。Q3/Q8 的 W 也慢，不能
把完整反向损失全归因于 B 或通信。mask 和旧时间在
`profile/qkv_backward_cp4_long_v1/scope_summary.json`。

24 个 auto 宏样本、80 个 c4/8/12/16/24 角色样本已完成（416 rank），
Graph、10 次预热、beta0 B→W、aggregate only。auto 一律 c12/N256C2。
Q3/Q8 路由约在计算角色包络的 18% 完成，Q2/Q7 约一半；Q4/Q5 更接近
平衡。计算包络包含输入等待；49,664 个 eligible compute CTA 的 first-ready
没有观测记录，null 不能记成零。128K/512K 进入校准，256K 留作验证。

Profiling OFF 的 12 点独立 B CTA A/B 完成：144 arms、7,200 rank-max、
28,800 raw rank 时间，配置/来源/正确性全部通过。Q3 c4 三长度 GM 为
1.03133/1.02566/1.03387×，Q8 为 1.04178/1.04078/1.03495×；Q2/Q7 c8
并不稳定，c4 明显退化。表保留全部候选，不将独立 B 当完整 B→W 收益。
旧 CTA 和 W 默认均未修改，未达四算子里程碑。

仅用 80 个角色 C/R 预览 `max(a·GFLOP/(132−c),tasks·(u/c+v))`，
其中 tasks=MK/(128×128)，计算拟合使用 observed R 避免把供数等待当裸计算。
训练 C/R MAPE 为 6.84%/8.25%，留一几何路由最大单点误差仍约 40.4%；
N4096 选择会在 c8/c10 间变化，N5120 的小幅预测收益也未稳定实证。
预览明确未冻结、不写生产。诊断 B→W 与正式孤立 B 的运行历史不同，
下一步须补完整边界验证及模型误差约束；W 缺口另行分析。
证据为 `profile/qkv_backward_cp4_{long,balance}_v1`、
`sweep/cp4_qkv_backward_compute_budget_v1{,_summary}.json` 和
`calibration/qkv_backward_cp4_role_model_preview_v1.json`，没有新增逐 shape 脚本。

完整边界补测复用原 sweep，新增 `--backward-phase total` 与正式配对反序选项。
beta1 的 B-only 与 W 通过和原正式 runner 共用的 `launch_boundary()` 顺序发起，
beta0 保持原生组合入口；不改内核或原默认计时行为。两个 4K 几何的两种 beta、
非零累加/覆盖和完整输出预检通过，296 项 CPU 检查通过。
Q3/Q8 三个长序列、Graph、两种 beta 的正序完整 B→W 已完成：72 arms、
3,600 rank-max/14,400 raw rank 时间，配置、FLOPs、统计和正确性审计通过，
平均 auto/c4=1.019043×。这是该六点的单序结果，不是完整 scope 或里程碑。
正反序现均已验收，源码/库相同，共 144 arms、7,200 rank-max / 28,800 raw
rank 时间；所有实际配置、统计、输出和非零 beta 覆盖/累加检查通过。
六轮配对比值等权 GM 为 **1.019534×**，12 个 case/mode 均为正，范围
1.015343–1.023646×。Graph capture 与短搜均保持 auto-first，仅正式执行顺序
反转。原始文件为 `sweep/cp4_qkv_backward_complete{,_reverse}_v1.json`；
双序汇总为 `sweep/cp4_qkv_backward_complete_balanced_v1_summary.json`。
生产 CTA 仍未修改；这不是完整 scope 或四算子里程碑，W 缺口仍未解决。

补齐 BF16/MXFP8 QKV B 的 first-ready 宏接线后，四几何两个长 T、c4/12/24
共 24 点及 Q8 六个反序点全部通过；14,240 个 compute CTA 均有首个 acquire
观测。512K 首轮最大 startup/compute 包络比例为 2.1062%，不是累计等待。
Q8 单次宏数据有较大波动，保留正反序全部摘要，不取最快值替代正式 A/B。
生产/参考中筛选的 10 个 QKV backward kernel 的 SASS 编码在改动前后及
ON/OFF 库间一致；不宣称已审计整个 DSO 的每个内核。正式策略不变。

短点诊断回归发现旧 reporter 把 SM 数当实际 grid，1K 两个几何实际只发射
74 CTA。现按已有静态 scheduler 的受支持 cluster1/2、AlongN/swizzle1 几何
推导 grid，并校验未使用 suffix 全零、已启动记录无缺失，不按非零计数蒙混。
1K/16K Eager beta1 四点通过，覆盖 N128/C1、N64/C2、N256/C2；本次新增
telemetry 类型其余 tile 只完成编译，不宣称这四点涵盖全部 tile 的实机验证。
298 项 CPU 检查通过。诊断/审计位于 `profile/qkv_backward_cp4_first_ready*`
和 `validation/qkv_backward_ready{,_grid}_v1`。
下一步仍只做 QKV Backward：补 Q3/Q8 匹配的裸计算/预发布 ready 服务参照，
区分固定计算实现的 CTA 分配空间与 GEMM/WGrad 本身缺口；资源参照不当硬上限。

上述参照现已完成：Q3/Q8 三个长 T、c4/c12、裸/ready subgrid 与裸 fullgrid，
36 项 10+50，1,800 rank-max / 7,200 raw-rank 样本全部审计通过。
ready/bare GM=1.000249，裸 120/128=1.054484，裸 128/132=1.001000。
这是单序原语比较，不是双序 A/B；同配置 132 CTA 控制也有最多约 2.33% 波动。
没有依据继续把大 N 的主要缺口归因 ready 指令，或假设 128→132 必有线性收益。

Q8/128K c4 beta0 的四卡 NSYS 诊断已采集并校验映射：B/W 各 13 条/卡，
排除 eager setup 后每组 11 条，指标内窗覆盖约 99.3%。W Tensor Active
各卡 96.26%–96.80%，DRAM 读 55.57%–57.29%；B Tensor Active 为
89.58%–93.86%。GPU0 最后一条 W=42.899ms、GPC=868.4MHz、Tensor=98.23%；
因此不能把单次变慢直接写成 kernel 调度瓶颈。没有锁频、改功耗或宣称因果。
W 时延与频率倒数在各卡的观测相关为 0.9981/0.9836/0.9391/0.9498，
不将 NSYS 下的 11 条诊断值冒充正式 50 样本或硬件上限。
数据为 `calibration/cp4_qkv_large_compute_v1{,_summary}.json` 和
`profile/qkv_backward_cp4_compute_metrics_v1/`，后者文档含完整复现与局限。
接下来回到冻结 bad mask 中尚无稳定方案的 Q2/Q7，先验证同完整 B→W 边界
的资源分配，避免用 B-only 与 B→W 不同历史解释模型误差；生产策略仍未改。

Q2/Q7 完整边界正序已完成：六个长点、Graph、beta0/1，72 arms、
3,600 rank-max / 14,400 raw-rank 样本，配置、统计、FLOPs、完整输出与非零
beta 检查均通过。auto(c12)/c8 GM=0.995826×，不支持把模型对 N5120 的
c8 预测提升为生产规则；Q7/128K beta0=0.952185×，其余点也没有普遍收益。
这是单序结果，不宣称完成双序验收。数据为
`sweep/cp4_qkv_backward_mid_complete_v1{,_summary}.json`。
下一步先解释这类模型误差：需要与正式 B→W 相同执行历史的角色/频率观察，
不能再用单次独立宏的甜点值替换配对结果，也不为此改 GPU 时钟/功耗。

第一轮通信 CTA 探索（尚未修改生产 auto）：固定 `production_qwen_dense` 几何、
全局 128K，CP4/CP8 四算子。短搜 2+7 只筛选；关闭 profiling 的候选 Graph
复用后进行三轮交错 10+50 A/B。八个 case 共 96 个正式测量分支，其 rank-max
向量与源码哈希已核对。保留在 `sweep/{cp4_four,cp8_other,cp8_oproj_forward}_v1.json`。

| 算子阶段 | CP4 相对 auto | CP8 相对 auto | 当前结论 |
|---|---|---|---|
| QKV F | c12：0.974–1.055× | c12：1.042–1.047× | 小收益需更广泛验证 |
| OProj F | c16：1.394–1.401× | c16：2.742–2.770× | 通信 CTA 不足的假设得到此几何的 A/B 支持 |
| QKV B | c24：1.524–1.549× | c24：1.567–1.582× | 此几何的 B 路径明确改善 |
| OProj B | c12：0.983–1.011× | c12：1.008–1.022× | 不据此更改默认配置 |

这里的反向只计 B，不计 W，不能当作完整反向或全矩阵里程碑。下一步扩大几何
与长度，并通过独立角色/资源测量拟合泛化模型；不把表内候选写成 shape winner 表。

第二轮增加 H=3072/4096/5120 的全局 128K、OProj F/QKV B，共 12 个 case，
同样三轮交错 10+50，结果在 `sweep/cp{4,8}_geometry_v1.json`。CP8 OProj F
的成对几何平均分别约 1.629/1.341/1.276×；QKV B 只有 H3072 的 c16 稳定
约 1.140×，H4096/5120 不支持统一增大 CTA。CP4 较多点的 auto 自身也有
明显漂移，暂不据此宣称小收益或拟合逐点最佳值。

当前正在验证的 FP32 W GEMM 候选仅改变通用 tile/cluster：基线
128×256×64/C2，候选 128×128×64/C2、128×128×128/C2、128×256×64/C1。
默认 auto 尚未改变；stage/共享内存通过真实模板约束，寄存器由
cudaFuncGetAttributes 获取，不能从 accumulator 尺寸推称占用率翻倍。
`operator_sweep.py --sweep-kind wgrad` 复用同一套短搜和交错 A/B，单独测试
beta0/1，并检查非零旧梯度覆盖与连续累加。W-only 结果不含 DQ/B/通信。

模型校准复用已有 OProj BF16 compute-subgrid/fullgrid 和 copy reference。
两种 compute 保持相同 tile，区别只是可用 SM 数；copy 需与融合路径共享
frontier-window 公式，原二参数 reference 的固定窗口不能被误称为同调度。
宏打点校准支持 `--aggregate-only`，每 shape 复用 IPC、timeline 与 marker，
只保存一个逐项 checkpoint 的摘要 JSON，不生成大批 Perfetto。

本轮实测：CP8 两种 W 几何、两算子、beta0/1 共八组完整交错 A/B 没有发现
稳定收益。N128/K64 约慢 2%–5%；C1 基本持平。N128/K128 的约两倍退化来自
2+7 短筛选，未进入正式重复 A/B，不能单独归因于 stages。默认 W 保持基线。

OProj 独立原语的 CP4/8 各 20 个校准点（四几何×五 CTA）、共 120 个原语测量
已完成 10+50 与独立正确性检查，保存 `calibration/cp{4,8}_primitives_v1.json`。
同调度 CP8 小 hidden 的 c4：copy 825.5µs / compute-subgrid 234.4µs；
c16：270.5/259.9µs；c24：236.0/285.3µs。通信服务改善和计算让出 SM 的代价
已被分开测出。宏打点增加 15 个紧凑样本，不把包络当独立服务时间。

CP8 留出长度 256K/512K 共八个 F/B case 的三轮 A/B 已完成：小 hidden
OProj F 的 c24 分别约 1.645/1.682×；QKV B 的 c32 约 1.749/1.843×；
H4096/K4096 的 QKV B 以 c16 约 1.130/1.140×。H4096 的 OProj F 则原 auto
c8 更好，c12/16 均未赢。更长 T 会改变甜点区，不能使用只看 hidden 的常量公式。
这些仍是逐 F/B 路径结果，不含完整反向 W，未更新四算子里程碑。

CP4 也补齐对应的八个长序列留出 case：小 hidden 的 OProj F 成对均值
约 1.26–1.41×，QKV B 的 c24 约 1.39–1.45×；H4096 的 F/B 均未证明
优于原 auto。其源码按 `5b98178` 核对，原始向量位于
`sweep/cp4_length_holdout_v1.json`。这些差异要求模型保留物理 tile/task wave
与卡组服务率，不能只把一个 hidden 对应到固定 CTA。

已否决的简化 QKV B 模型：仅用 FLOPs/(SM−c) 与 remote-bytes/c 的 max，
需要约 339µs 自由常数才能拟合 128K 曲线；它不是真实 launch/DQ 时间。
限制常数并加入部分重叠后，留一几何误差仍约 12%，不足以落默认策略。
后续以独立服务、整数 tile waves、实际 route task waves 和流水填充项继续检验。

OProj F 的独立服务拟合已固化为 `operator_model.py`，输出显式、冻结的
`calibration/oproj_service_model_v1.json`。只用原语数据拟合非负计算/拷贝服务率，
不使用 fused winner；整数 tile/task waves 与 frontier window 决定候选分数。
固定 5µs 是模型先验而非测得的 launch，填充项仍是待验证假设。生产默认不变；
实验参数 `--oproj-comm-model` 必须显式指定卡组校准，并要求至少 10% 预测时间
下降才切换。原生/Python 选择一致性现已通过：204 个 OProj/布局/域检查，
以及 288 个其他算子不受影响检查；这是单卡 metadata 查询，不是多卡计算证据。
另一次默认路径 16 个多卡算子正确性检查通过。实际模型策略采用 CP4/8 全 OProj
矩阵三轮 Eager/Graph A/B，不用计时结果重选或重拟合参数。

上述 A/B 已完成正序和反序的全量平衡验收：CP4/8 各 48 个原 setting、
每 setting 两种 launch、每顺序三轮 10+50。合计 115,200 个 rank-max 样本，
691,200 个逐 rank 时延，全部正确性与来源/配置/向量审核通过。
未换配置的 CP4 长点在正序/反序曾呈相反方向的小幅变化，因此最终固定两顺序
等权，不挑某次重测覆盖旧点。模型系数没有用这些结果重拟合。

| OProj F / 对原自动策略 | 全量 GM | 长序列 GM | 长序列未换配置 GM |
|---|---:|---:|---:|
| CP4 | 1.04038× | 1.04019× | 1.00089× |
| CP8 | 1.08857× | 1.16183× | 1.00014× |
| 合并 | 1.06420× | 1.09933× | — |

全量最差时延退化 1.33%，长序列最差 1.13%，均在未换配置的点；这些是观测
结果而非“统计显著性”保证。完整表在 `policy_ab/summary/policy_summary.*`，
保留全部回退与退化及正反序分项。它只证明 OProj F 的显式校准策略，生产默认
仍未改，也不是相对 TEUB/cuBLASLt 的四算子里程碑。该轮 CPU 检查 214 项通过。

冻结候选库仅保留一份 `policy_ab/libfuse_mxfp8_frozen_model.so`（6,432,744 bytes，
SHA256 `cfba324975f1bb9d5568d9a8a01dfe7c864594d6ffdf815acae37a6cb18b9445`），
与四份正式 A/B 原始文件中的库哈希一致；不为每轮创建二进制快照。
QKV B 已区分裸计算、预发布 ready 的计算和 push-only copy，保留原始
RowMajor-B、head 粒度 system-acquire、真实 12-slot 路由和同一 scheduler。
两种 compute 同为 M128N256K64/C2、4 stages、214016 B 动态共享内存、
168 registers/thread；fullgrid 只改变 SM 预算，不重新选 tile。预发布与裸计算
的差值称为 `adapter_overhead_us`，不是通信等待。

CP4/8 各四个实际 (N,K) 几何、六档 c=4/8/12/16/24/32，共 48 个配置、
144 项独立原语，已完成 profiling OFF 的 10+50 和正确性检查，见
`calibration/cp{4,8}_qkv_primitives_v1.json`。每个 shape 复用 IPC/分配；
fullgrid 只在初始两点 pilot 测量，没有在六档相同 tile 下重复测六次。
`qkv_service_model.py` 严格核对原始向量、资源和物理工作量，不写 dispatch。

按 (N,K) 留一几何，整数 cluster waves × (a·tile TFLOPs + e·tile area)
加固定 5µs 先验的 compute 模型，bare/预发布 ready 的 MAPE 为
CP4 4.41%/4.35%、CP8 1.21%/1.54%。CP8 未参与拟合的 K6144 pilot
四个 subgrid/fullgrid 预测误差均小于 0.7%。这里只验证了 128K 和当前 tile，
不是全部长序列泛化；5µs 也不是实测 launch latency。

原独立 copy 的 `5+hypot(b·remote MiB,t·ceil(tasks/(12*c)))` 模型已被否决：
LOGO MAPE 为 CP4 19.11%、CP8 25.23%。CP4 K8192 的 c24/c32 独立
copy 为 1256.94/3098.00µs，逆序 copy-only 对照仍为 1252.69/3183.55µs；
而同几何融合宏打点 route 的 all-rank max 仅为 1283.01/1320.96µs。
这些边界不能混作一个服务时间，更不能把异常删除后拟合逐点 winner。

因此增加 `copy_fused_reservation` 对照：同一 copy entry、384 threads、
38 registers/thread、12 slots，但按 fused 类型使用 C2/214016 B；旧 copy
仍为 C1/196704 B。两条都没有并发 compute 或 finalize，不预先假定新对照
就等价 fused route。新对照的 CP4/8 各 24 点已通过严格 10+50 和正确性检查，
保存在 `calibration/cp{4,8}_qkv_copy_reservation_v1.json`。相同通信工作量的
N 差异明显收敛，但模型 LOGO MAPE 仍为 16.81%/21.30%，暂不写策略。
一个直接反例是 c4：K5120 比 K4096 多 25% payload/tasks，copy 时间却只有
其约 0.93–0.94 倍；正系数的 payload/task-depth 单调模型无法解释此关系。
后续以宏测量并发 compute 条件下的 route 包络，不把它改称孤立 NVLink 带宽。
QKV 生产自动策略尚未修改，四算子阶段目标仍未达到。

新版全量采用显式冻结的 OProj F 模型，其他三个算子的自动策略保持不变。
启动器、QKV 原语分析与计算参考报告的 CPU 检查合计 242 项通过；本轮不追加
GPU 搜索或多轮 A/B，先完成 CP4/CP8 的一组正式全量再选择下一处优化。

该组全量已完成并通过 `d89e40e` 源码/库哈希、配置、10+50 原始 rank-max
向量及正确性审计：384 settings、2304 records、1152 个完整边界，无缺项。
`operators/full_model_v1/comparison` 保留与旧格式一致的全量宽表和摘要。
全量边界等权 / 四算子等权为 1.038383× / 1.046918×；长序列为
1.151840× / 1.169852×，463/576 获胜，1.2× 第一阶段尚未达到。
同轮纯 GEMM 的长序列参考为 1.329338× / 1.350255×；仍假设通信和 DQ
均免费，不是硬件上限。没有覆盖初版、重搜外部基线或新增 GPU 测试。

该次全量发现的反向薄弱项是 CP8 `production_qwen_dense` QKV B+W：
128K/256K/512K 的 launch/mode 等权 GM 分别为 0.922449× / 0.930020× /
0.928707×。最弱 128K Graph beta0 为基线 1412.640µs、Fuse 1550.320µs，
比例 0.911193×。尚不把这组完整边界损失全部归因通信；需结合已有 B/W 与
宏测量再判断。按用户要求先展示全量效果，本轮到此结束，不追加调优。

W 追加的 N192/K64/C2（4 stages）与 N256/K32/C2（6 stages）已完成
CP4/8 的 32 组正确性与 65,536 对量化编码检查，全部通过；实际寄存器仍为
168/thread。CP8 两几何、两算子、beta0/1 的三轮 10+50 A/B 位于
`sweep/cp8_wgrad_extended_v1.json`：K32 基本持平（约 0.992–1.014×），
N192 在小 hidden 的 QKV/OProj W 分别约 0.89/0.69×，大 hidden 也无稳定收益。
这些数据不支持仅增加 stages 的优化假设，W 默认仍保留 N256/K64/C2。
该轮 CPU 检查 170 项通过；上述结果均不是全量 2F2B 里程碑证据。

沿用仓库 `PROFILE_PROTOCOL.md`、`include/fuse/profiling/timeline.cuh`
和 `FUSE_ENABLE_PROFILING`。先分开观察运行时 DQ、B/F 的 compute/route/ready
等待/finalize，以及纯 W GEMM。包络和等待不能被误称为裸 Tensor Core 或 NVLink 时间；
不能跨 GPU 相减绝对 globaltimer，也不能把相互重叠的等待简单相加。

profile 构建仅作诊断。正式收益必须在 profiling 关闭、10 warmup / 50 次
sample-wise rank-max 的完整边界中复测。分配、Graph capture、JIT、调优、通信器
初始化均不进入计时；跨 shape 复用进程，已有基线不重复全量搜索。

逐个算子确定已测瓶颈，再检查 GEMM tile 和通信 CTA 的性能甜点区。参考既有
SM90 BF16/FP8 的独立 GEMM wave、route 原语和资源测量，拟合仅依赖硬件、MNK、
CP/head 几何、tile/cluster、工作量和资源的模型。配置与模型版本写入正式结果。
保留未用于校准的合法几何做泛化验证。

## 红线

- 不按模型名、特定 shape 或逐 benchmark winner 写特判/查表；
- 有可测证据时必须测量；记录明确区分“已测事实”“待验证推测”和“已否定假设”；
- MXFP8 的每 32 元素 E8M0 scale 语义不变，不靠改变 block size、精度或跳过 DQ 提速；
- 可测量探索反量化的向量化、加载和流水安排，但不能偷缓存固定权重后漏计调用成本；
- 激活/通信/data gradient 仍 BF16，FP32 dW 仍匹配 beta=0/1；
- 不改原有 BF16/FP8 行为，不带 KDA，不加入训练 batch/优化器/模型 E2E 范围；
- 不把诊断 trace 当正式性能，不隐藏失败、缺失点、回退或观察开销。

## 结果保留与清理

每轮优化结束清理一次。本任务只保留正式初版基线、当前最佳正式结果、正在验证
的一轮候选，以及支撑关键结论的少量 profile。旧候选只保留紧凑结论、参数和源码
版本；重复 trace、可重建的冗余阶段表、失败尝试的临时脚本不长期保留。
可复用的采样/验证脚本继续保留，不为每个 shape 创建一个脚本。

默认仅生成原有风格的 `comparison_summary.{json,csv,md}` 与 metadata；如需临时
阶段明细，用 `--phase-details` 重建，看完清理。全量正式结果完成后，小规模 pilot
只保留覆盖/通过摘要，删除被正式结果替代的逐样本数据。不清理运行中的文件，
不动原工作树或本任务之外的数据。删除前核对确切路径和可重建方式，清理后记录。
大生成文件删除只执行精确路径的删除命令并报告路径/大小，不展开文件内容或大段
删除 diff，避免终端与会话被无意义的输出占满。

first-ready 回归完成后已删除无有效样本的失败 checkpoint、四个短点的原始
诊断时长及被 298 项检查替代的 296 项 CPU 日志/manifest，共 101,349 bytes；
两处空 profile 目录也移除。正确性、实际 grid/config、来源、删除前 SHA 与失败
原因保留在 `validation/qkv_backward_ready_grid_v1/gpu_checks.json`。
原短诊断时长不再保留，可重跑；30 个长点诊断摘要、正式
B→W A/B 向量和 SASS 编码审计保留。没有新建逐 shape 脚本或 SASS dump 文件。

Q7 状态诊断完成后已删除 `/tmp/fuse-q7-metrics.3mRZPg/` 中两份 SQLite
导出与 export.log，共 118,603,685 bytes；空目录同时移除。导出可从保留的
两份 NSYS 原报告重建，未删除原始计时/正确性或正式 A/B。未保存 SASS dump，
只在诊断文档保留类型、资源与编码哈希。

固定通信预算的长序列 tile 实测完成后，4K 预检仅保留 12 个配置的正确性、
来源哈希、设备与复现参数，移除 `sweep/cp4_qkv_tile_check_v1.json` 的短样本
原数据 80,534 bytes。摘要为同目录 `_summary.json`；可重跑，但原始短计时
不再保留。正式六点原始向量完整保留，没有生成新的 profile 或 SASS 文件。

完整 B→W 正序通过后，4K 预检保留八组配置/正确性（含非零 beta 检查）及
来源摘要，删除 `sweep/cp4_qkv_backward_total_check_v1.json` 的原始短计时
97,495 bytes。摘要为同名 `_summary.json`；短计时可重跑，不覆盖正式长数据。

本轮已删除被新校准替代的旧 `profile/candidate/` 两个 JSON，共 330,049 bytes。
未保留其原始时间戳，需复现时用现有脚本重新采集；初版八个 profile、初版全量
基线和当前 aggregate 校准均保留。没有生成新的逐 tile 文件或专用 shape 脚本。

双序全量验收完成后，已删除被完整覆盖的 `policy_ab/cp4_variance_check_v1.json`
及 `policy_ab/cp4_reverse_control_v1.json`，共 2,027,479 bytes。删除前确认各六点
均在相应全量文件中，且模型、库、几何、配置一致；其原始采样值不再保留，可用
同一 runner 重新测量。最终只保留四份正式全量原始文件和一套紧凑双序汇总。

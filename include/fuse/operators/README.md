# Operator header layout

Read the headers in this order:

1. `primitives/gemm_a2a.h` and `primitives/a2a_gemm.h` define the two physical
   dataflow orders and their parameter storage.
2. `ulysses/projection_dataflow.h` is the semantic registry. It maps QKV or
   output projection plus forward or backward onto one dataflow primitive.
3. `ulysses/qkv_backward.h` and `ulysses/oproj_backward.h` add the independent
   weight-gradient phase and the immediate/deferred ZeroBubble contract.
4. `ulysses/heterogeneous_cp.h` contains the optional weighted-CP planner.

The flat headers in this directory are compatibility includes. New code should
include the canonical path above. A new projection dataflow must be registered
in `projection_dataflow.h`; a new kernel tile must be registered once in
`csrc/operators/ulysses_sm90/detail/launch.cuh`.

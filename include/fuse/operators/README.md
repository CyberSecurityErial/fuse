# Operator header layout

Read the headers in this order:

1. `primitives/gemm_a2a.h` and `primitives/a2a_gemm.h` define the two physical
   dataflow orders and their parameter storage.
2. `semantics/operator.h` defines the model-independent compile-time contract
   used to attach layout, route and completion meaning to a physical dataflow.
3. `semantics/ulysses/projection.h` is the Ulysses semantic registry. Its four
   specifications declare projection/pass, input/output layouts, route,
   completion rule, precision parameter types and public launch entry points.
   Kernel bindings consume the whole specification rather than reconstructing
   semantics from alias names.
4. `ulysses/qkv_backward.h` and `ulysses/oproj_backward.h` add the independent
   weight-gradient phase and the immediate/deferred ZeroBubble contract.
5. `ulysses/heterogeneous_cp.h` contains the optional weighted-CP planner.

The flat headers and `ulysses/projection_dataflow.h` are compatibility
includes. New code should include the canonical paths above. A change to
tensor meaning belongs in one semantic specification; a new kernel tile must
be registered once in `csrc/operators/ulysses_sm90/detail/launch.cuh`.

Do not put model or projection branches into a primitive. A primitive may use
the specification's declared dataflow contract at compile time, while runtime
request scheduling and device placement remain outside the operator kernel.
Projection/pass, precision variants and weight-gradient policy are a Ulysses
extension of the generic contract, not requirements imposed on every future
semantic operator.

#!/usr/bin/env python3
"""Host execution of the real benchmark Graph helper with explicit CUDA mocks.

These tests establish API sequencing and fail-closed state transitions, not
CUDA Graph support, cooperative scheduling, IPC visibility or GPU correctness.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FusedGraphContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("host C++ compiler required")
        temporary = tempfile.TemporaryDirectory(prefix="fuse-graph-contract-")
        cls.addClassCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        header = r'''
#pragma once
#include <cstddef>
#include <cstdint>
using cudaError_t=int;
using cudaStream_t=void*;
constexpr int cudaSuccess=0, cudaErrorNotReady=34, cudaErrorInvalidValue=1;
struct dim3 { unsigned int x,y,z; constexpr dim3(unsigned int a=1,unsigned int b=1,unsigned int c=1):x(a),y(b),z(c){} };
enum cudaStreamCaptureMode { cudaStreamCaptureModeThreadLocal=1 };
enum cudaGraphNodeType { cudaGraphNodeTypeKernel=0, cudaGraphNodeTypeMemcpy=1 };
enum cudaGraphExecUpdateResult { cudaGraphExecUpdateSuccess=0, cudaGraphExecUpdateError=1 };
enum cudaKernelNodeAttrID { cudaKernelNodeAttributeCooperative=2,
  cudaKernelNodeAttributeClusterDimension=4, cudaKernelNodeAttributeClusterSchedulingPolicyPreference=5 };
union cudaKernelNodeAttrValue {
  int cooperative;
  struct { unsigned int x,y,z; } clusterDim;
  int clusterSchedulingPolicyPreference;
};
struct Graph;
struct Executable;
using cudaGraph_t=Graph*;
using cudaGraphNode_t=Graph*;
using cudaGraphExec_t=Executable*;
struct cudaKernelNodeParams {
  void* func; dim3 gridDim, blockDim; unsigned int sharedMemBytes; void** kernelParams; void** extra;
};
struct cudaGraphExecUpdateResultInfo {
  cudaGraphExecUpdateResult result;
  cudaGraphNode_t errorNode, errorFromNode;
};
cudaError_t cudaGetDevice(int*);
cudaError_t cudaStreamQuery(cudaStream_t);
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaStreamBeginCapture(cudaStream_t,cudaStreamCaptureMode);
cudaError_t cudaStreamEndCapture(cudaStream_t,cudaGraph_t*);
cudaError_t cudaGraphGetNodes(cudaGraph_t,cudaGraphNode_t*,size_t*);
struct cudaGraphEdgeData { unsigned char from_port=0,to_port=0,type=0,reserved[5]{}; };
cudaError_t cudaGraphGetEdges(cudaGraph_t,cudaGraphNode_t*,cudaGraphNode_t*,cudaGraphEdgeData*,size_t*);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t,cudaGraphNodeType*);
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t,cudaKernelNodeParams*);
cudaError_t cudaGraphKernelNodeGetAttribute(cudaGraphNode_t,cudaKernelNodeAttrID,cudaKernelNodeAttrValue*);
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t*,cudaGraph_t,unsigned long long);
cudaError_t cudaGraphUpload(cudaGraphExec_t,cudaStream_t);
cudaError_t cudaGraphExecUpdate(cudaGraphExec_t,cudaGraph_t,cudaGraphExecUpdateResultInfo*);
cudaError_t cudaGraphLaunch(cudaGraphExec_t,cudaStream_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
'''
        (directory / "cuda_runtime.h").write_text(header)
        source = r'''
#include "benchmarks/sm103/fused_graph.cuh"
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#define CHECK(value) do { if(!(value)) throw std::runtime_error(#value); } while(0)
struct Graph {
  size_t count=1;
  cudaGraphNodeType type=cudaGraphNodeTypeKernel;
  void* function=reinterpret_cast<void*>(0x1000);
  dim3 grid{148,1,1}, block{256,1,1}, cluster{0,0,0};
  unsigned int smem=230400;
  int cooperative=1, scheduling=0;
  uint32_t epoch=0;
  void* argument=nullptr;
};
struct Executable { uint32_t epoch=0; };
cudaStream_t stream=reinterpret_cast<void*>(0x2000);
std::string fault;
Graph next;
Graph post;
int owner_device=0, begins=0, ends=0, instantiates=0, uploads=0, updates=0, launches=0;
int graph_live=0, exec_live=0;
bool capturing=false, busy=false;
std::vector<uint32_t> executed;
int cudaGetDevice(int* device) { *device=owner_device; return 0; }
int cudaStreamQuery(cudaStream_t s) { CHECK(s==stream); return busy?cudaErrorNotReady:cudaSuccess; }
const char* cudaGetErrorString(cudaError_t error) { return error?"mock CUDA failure":"cudaSuccess"; }
int cudaStreamBeginCapture(cudaStream_t s,cudaStreamCaptureMode mode) {
  CHECK(s==stream && mode==cudaStreamCaptureModeThreadLocal && !capturing && !busy);
  ++begins;
  if(fault=="begin") return 1;
  capturing=true; return 0;
}
int cudaStreamEndCapture(cudaStream_t s,cudaGraph_t* out) {
  CHECK(s==stream && capturing); capturing=false; ++ends;
  *out=new Graph(next); (*out)->argument=&(*out)->epoch; ++graph_live;
  return fault=="end"?1:0;
}
int cudaGraphGetNodes(cudaGraph_t graph,cudaGraphNode_t* nodes,size_t* count) {
  CHECK(graph);
  if(fault=="get_nodes") return 1;
  *count=graph->count;
  if(nodes && graph->count) { nodes[0]=graph; if(graph->count==2) nodes[1]=&post; }
  return 0;
}
int cudaGraphGetEdges(cudaGraph_t graph,cudaGraphNode_t* from,cudaGraphNode_t* to,cudaGraphEdgeData* data,size_t* count) {
  *count=fault=="unordered"?0:1;
  if(from) *from=fault=="reverse_edge"?&post:graph;
  if(to) *to=fault=="reverse_edge"?graph:&post;
  if(data) data->type=fault=="programmatic_edge"?1:0;
  return 0;
}
int cudaGraphNodeGetType(cudaGraphNode_t node,cudaGraphNodeType* type) { *type=node->type; return 0; }
int cudaGraphKernelNodeGetParams(cudaGraphNode_t node,cudaKernelNodeParams* params) {
  if(fault=="get_params") return 1;
  *params={node->function,node->grid,node->block,node->smem,&node->argument,nullptr}; return 0;
}
int cudaGraphKernelNodeGetAttribute(cudaGraphNode_t node,cudaKernelNodeAttrID id,cudaKernelNodeAttrValue* value) {
  if(fault=="get_attribute") return 1;
  if(id==cudaKernelNodeAttributeCooperative) value->cooperative=node->cooperative;
  else if(id==cudaKernelNodeAttributeClusterDimension) value->clusterDim={node->cluster.x,node->cluster.y,node->cluster.z};
  else { CHECK(id==cudaKernelNodeAttributeClusterSchedulingPolicyPreference); value->clusterSchedulingPolicyPreference=node->scheduling; }
  return 0;
}
int cudaGraphInstantiateWithFlags(cudaGraphExec_t* out,cudaGraph_t graph,unsigned long long flags) {
  CHECK(flags==0 && graph->epoch==next.epoch); ++instantiates;
  if(fault=="instantiate") return 1;
  *out=new Executable{graph->epoch}; ++exec_live; return 0;
}
int cudaGraphUpload(cudaGraphExec_t,cudaStream_t s) {
  CHECK(s==stream); ++uploads;
  if(fault=="upload") return 1;
  busy=true; return 0;
}
int cudaGraphExecUpdate(cudaGraphExec_t exec,cudaGraph_t graph,cudaGraphExecUpdateResultInfo* result) {
  CHECK(!busy && graph->epoch==next.epoch); ++updates;
  result->result=fault=="update_result"?cudaGraphExecUpdateError:cudaGraphExecUpdateSuccess;
  if(fault=="update_status") return 1;
  if(result->result==cudaGraphExecUpdateSuccess) exec->epoch=graph->epoch;
  return 0;
}
int cudaGraphLaunch(cudaGraphExec_t exec,cudaStream_t s) {
  CHECK(s==stream && !capturing);
  if(fault=="launch") return 1;
  executed.push_back(exec->epoch); ++launches; busy=true; return 0;
}
int cudaGraphExecDestroy(cudaGraphExec_t exec) { delete exec; --exec_live; return 0; }
int cudaGraphDestroy(cudaGraph_t graph) { delete graph; --graph_live; return 0; }
int actual_launch(uint32_t epoch,cudaStream_t s) {
  CHECK(capturing && s==stream); next.epoch=epoch;
  if(fault=="callback_throw") throw std::runtime_error("callback exception");
  return fault=="callback_status"?1:0;
}
void complete() { busy=false; }
template<class F> void rejects(F&& f) {
  bool rejected=false;
  try { f(); } catch(const std::exception&) { rejected=true; }
  CHECK(rejected);
}
void alter(const std::string& field) {
  if(field=="empty") next.count=0;
  else if(field=="multi") next.count=2;
  else if(field=="memcpy") next.type=cudaGraphNodeTypeMemcpy;
  else if(field=="cooperative") next.cooperative=0;
  else if(field=="cluster") next.cluster={2,1,1};
  else if(field=="partial_cluster") next.cluster={1,0,0};
  else if(field=="explicit_cluster") next.cluster={1,1,1};
  else if(field=="scheduling") next.scheduling=1;
  else if(field=="block") next.block={128,1,1};
  else if(field=="grid") next.grid={147,1,1};
  else if(field=="grid_y") next.grid={148,2,1};
  else if(field=="smem") next.smem+=128;
  else if(field=="zero_smem") next.smem=0;
  else if(field=="null_function") next.function=nullptr;
  else if(field=="function") next.function=reinterpret_cast<void*>(0x3000);
  else throw std::runtime_error("unknown mutation");
}
int main(int argc,char** argv) {
  CHECK(argc>=2); const std::string mode=argv[1];
  try {
    if(mode=="normal") {
      fused_graph::Operation op(0,stream,17);
      op.prepare(18,actual_launch);
      CHECK(op.committed_epoch()==17 && launches==0 && instantiates==1 && uploads==1);
      CHECK(graph_live==1 && exec_live==1 && !capturing);
      complete(); op.launch(); CHECK(op.committed_epoch()==18 && executed.back()==18);
      rejects([&]{op.prepare(19,actual_launch);}); // Previous sample is not complete.
      complete(); op.prepare(19,actual_launch);
      CHECK(updates==1 && graph_live==1 && instantiates==1 && uploads==1);
      CHECK(op.committed_epoch()==18); op.launch(); CHECK(executed.back()==19);
      complete(); op.reset(19); CHECK(graph_live==0 && exec_live==0);
      op.prepare(20,actual_launch); complete(); op.launch(); complete(); op.reset(20);
      CHECK(instantiates==2 && op.committed_epoch()==20 && executed==std::vector<uint32_t>({18,19,20}));
    } else if(mode=="separate" || mode=="separate_qkv") {
      const bool qkv=mode=="separate_qkv";
      next.count=2; post.cooperative=qkv?1:0; post.smem=qkv?0:16448; post.grid={16384,1,1};
      post.function=reinterpret_cast<void*>(0x4000);
      fused_graph::Operation op(0,stream,0,true,qkv);
      if(argc==3) {
        fault=argv[2];
        if(fault=="bad_post") post.cooperative=qkv?0:1;
        rejects([&]{op.prepare(1,actual_launch);});
      } else {
        op.prepare(1,actual_launch); complete(); op.launch(); complete();
        op.prepare(2,actual_launch); op.launch(); complete();
        CHECK(launches==2 && updates==1 && op.committed_epoch()==2);
        post.smem+=16;
        rejects([&]{op.prepare(3,actual_launch);});
      }
      op.reset(op.committed_epoch());
    } else if(mode=="epochs") {
      fused_graph::Operation op(0,stream);
      rejects([&]{op.prepare(0,actual_launch);}); rejects([&]{op.prepare(2,actual_launch);});
      CHECK(begins==0); op.prepare(1,actual_launch); complete();
      rejects([&]{op.prepare(1,actual_launch);}); rejects([&]{op.prepare(2,actual_launch);});
      op.launch(); rejects([&]{op.launch();}); complete();
      rejects([&]{op.prepare(1,actual_launch);}); CHECK(launches==1 && begins==1);
      op.reset(std::numeric_limits<uint32_t>::max());
      rejects([&]{op.prepare(0,actual_launch);}); CHECK(begins==1);
    } else if(mode=="owner") {
      rejects([&]{fused_graph::Operation op(-1,stream);});
      rejects([&]{fused_graph::Operation op(0,nullptr);});
      fused_graph::Operation op(0,stream);
      owner_device=1; rejects([&]{op.prepare(1,actual_launch);}); CHECK(begins==0);
      owner_device=0; op.prepare(1,actual_launch); complete();
      owner_device=1; rejects([&]{op.launch();}); rejects([&]{op.reset();});
      owner_device=0; op.launch(); rejects([&]{op.reset();});
      CHECK(exec_live==1 && graph_live==1); complete(); op.reset(1);
    } else if(mode=="invalid" || mode=="changed") {
      CHECK(argc==3); fused_graph::Operation op(0,stream);
      if(mode=="changed") { op.prepare(1,actual_launch); complete(); op.launch(); complete(); }
      alter(argv[2]); const uint32_t epoch=mode=="changed"?2:1;
      rejects([&]{op.prepare(epoch,actual_launch);});
      CHECK(!capturing && updates==0 && op.committed_epoch()==epoch-1);
      rejects([&]{op.launch();}); rejects([&]{op.prepare(epoch,actual_launch);});
      complete(); op.reset(epoch-1);
    } else if(mode=="failure") {
      CHECK(argc==3); fused_graph::Operation op(0,stream);
      const std::string failure=argv[2];
      const bool update=failure=="update_status" || failure=="update_result";
      if(update) { op.prepare(1,actual_launch); complete(); op.launch(); complete(); }
      fault=failure;
      if(failure=="launch") {
        op.prepare(1,actual_launch); complete(); rejects([&]{op.launch();});
      } else rejects([&]{op.prepare(update?2:1,actual_launch);});
      CHECK(!capturing && op.committed_epoch()==uint32_t(update));
      CHECK(launches==int(update)); rejects([&]{op.launch();});
      if(failure=="begin") CHECK(ends==0); else CHECK(ends==begins);
      complete(); fault.clear(); op.reset(update?1:0);
      CHECK(graph_live==0 && exec_live==0);
      op.prepare(update?2:1,actual_launch); complete(); op.launch(); complete(); op.reset();
    } else if(mode=="destructor") {
      { fused_graph::Operation op(0,stream); op.prepare(1,actual_launch); complete(); op.launch(); complete(); }
    } else throw std::runtime_error("unknown scenario");
    CHECK(graph_live==0 && exec_live==0 && !capturing);
    std::cout<<"PASS "<<mode<<'\n';
  } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
'''
        path = directory / "graph.cpp"
        path.write_text(source)
        cls.probe = directory / "graph"
        result = subprocess.run([*compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
            "-I", str(directory), "-I", str(ROOT), str(path), "-o", str(cls.probe)],
            text=True, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stderr)

    def run_probe(self, *arguments):
        result = subprocess.run([str(self.probe), *arguments], text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_never_executes_and_replay_commits_contiguous_epochs(self):
        self.run_probe("normal")

    def test_separate_reference_is_one_ordered_two_kernel_graph(self):
        for mode in ('separate','separate_qkv'):
            self.run_probe(mode)
            for fault in ('unordered','reverse_edge','bad_post','programmatic_edge'):
                with self.subTest(mode=mode,fault=fault):
                    self.run_probe(mode,fault)

    def test_duplicate_skipped_zero_wrapped_and_unlaunched_epochs_are_rejected(self):
        self.run_probe("epochs")

    def test_owner_device_stream_completion_and_reset_contracts(self):
        self.run_probe("owner")
        self.run_probe("destructor")

    def test_non_kernel_non_cooperative_and_invalid_cluster_or_geometry_are_rejected(self):
        for field in ("empty", "multi", "memcpy", "cooperative", "cluster", "partial_cluster",
                      "block", "grid_y", "zero_smem", "null_function"):
            with self.subTest(field=field):
                self.run_probe("invalid", field)

    def test_update_preserves_function_grid_block_smem_and_cluster_attributes(self):
        for field in ("function", "grid", "block", "smem", "explicit_cluster", "scheduling", "cooperative"):
            with self.subTest(field=field):
                self.run_probe("changed", field)

    def test_capture_instantiation_upload_update_and_launch_failures_never_commit(self):
        for stage in ("begin", "callback_status", "callback_throw", "end", "get_nodes", "get_params",
                      "get_attribute", "instantiate", "upload", "update_status", "update_result", "launch"):
            with self.subTest(stage=stage):
                self.run_probe("failure", stage)

    def test_helper_is_not_enabled_by_the_default_harness(self):
        harness = (ROOT / "benchmarks/sm103/fused_bf16.cu").read_text()
        self.assertIn('#if FUSE_BENCH_MPI\n#include "fused_graph.cuh"\n#endif', harness)
        self.assertIn('std::string launch = "eager";', harness)
        self.assertIn('Graph requires the MPI target and excludes profiling/epilogue diagnostics', harness)


if __name__ == "__main__":
    unittest.main()

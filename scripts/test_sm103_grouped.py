"""Execute production grouped tile arithmetic on CPU; not a CUDA validation."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import re

ROOT = Path(__file__).resolve().parents[1]


class GroupedScheduleTests(unittest.TestCase):
    def test_prepare_window_and_actual_rows_properties(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        private = ROOT/'csrc/operators/sm103/detail/grouped'
        schedule = '\n'.join(line for line in
            (private/'producer_consumer.cuh').read_text().splitlines()
            if not line.startswith(('#include', '#pragma once')))
        communication = (private/'communication.cuh').read_text()
        policies = communication[communication.index('struct GroupedPrepareParams'):
                                 communication.index('// Fixed-size Graph node')]
        api = (ROOT/'csrc/operators/sm103/api/grouped.cuh').read_text()
        # Execute the production host initializer as well as device helpers:
        # a correct decoder alone cannot catch a missing window in the API.
        initialization = '\n'.join(re.search(pattern, api, re.S)[0] for pattern in (
            r'const int buffer_m = .*?;', r'g\.scheduler\.window_m = .*?;',
            r'GroupedTileOrder order\{.*?;')).replace('TileM','128')
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
namespace cute { template<class... T> struct Shape {}; }
''' + schedule + '\nnamespace fuse::detail { constexpr int kMaxWorldSize=8;\n' + policies + r'''
}
using namespace fuse::detail;
struct HostParams { int experts,dispatch_buffer_rows,expert_row_capacity; };
struct HostPolicy { int swizzle; bool along_n; };
GroupedTileOrder host_order(const HostParams& p, const HostPolicy& policy,
    const int64_t* tiles, int n_tiles) {
  struct { struct { int window_m=0; } scheduler; } g;
''' + initialization + r'''
  if(order.window_m!=g.scheduler.window_m)
    throw std::runtime_error("API prepare/communication order differs from GEMM window");
  return order;
}
struct Invocation {
  int compute_ctas=0;
  struct { struct { GroupedTileOrder order; } params; } comm;
};
struct UnusedSelector {};
void require(bool value) {
  if(!value) throw std::runtime_error("prepare property failed");
}
// Enumerate raster bands independently; never call decode/linear to establish
// the expected first wave. Expert IDs may move when empty experts are inserted.
int oracle(const std::vector<int>& rows,int n,int sw,bool along,int window,int workers) {
  std::vector<int> seen;
  int tasks=0,panels=0,base=0;
  for(int count:rows) {
    const int mt=(count+127)/128,step=window?window:std::max(1,mt);
    seen.resize(base+mt);
    for(int start=0;start<mt;start+=step) {
      const int width=std::min(step,mt-start);
      const int outer=along?width:n,inner=along?n:width;
      for(int band=0;band<outer;band+=sw)
        for(int major=0;major<inner;++major)
          for(int minor=band;minor<std::min(band+sw,outer);++minor) {
            if(tasks++>=workers) return panels;
            const int panel=base+start+(along?minor:major);
            if(!seen[panel]++) ++panels;
          }
    }
    base+=mt;
  }
  return panels;
}
void check_replays(std::vector<int> rows,int n,int sw,bool along,int buffer,int workers) {
  // All three replays retain the same storage and policy/order objects.
  std::vector<int64_t> offsets(rows.size()+1),tiles(rows.size()+1);
  const auto* original_rows=offsets.data();
  const auto* original_tiles=tiles.data();
  const int capacity=4096,window=buffer && buffer<capacity?buffer/128:0;
  auto order=host_order({int(rows.size()),buffer,capacity},{sw,along},tiles.data(),n);
  require(order.window_m==window);
  GroupedPrepareParams p{};
  p.row_offsets=offsets.data(); p.experts=int(rows.size());
  p.order=order; p.num_compute_ctas=workers;
  Invocation invocation{workers,{{order}}};
  GroupedPreparePolicyPatch<Invocation,UnusedSelector> dynamic{};
  dynamic.invocation_params=&invocation;
  GroupedExplicitPreparePolicy explicit_policy;
  for(int replay=0;replay<3;++replay) {
    if(replay==1) std::fill(rows.begin(),rows.end(),0);
    if(replay==2) for(size_t e=0;e<rows.size();++e) rows[e]=(e%2)?192:191;
    for(size_t e=0;e<rows.size();++e) {
      offsets[e+1]=offsets[e]+rows[e];
      tiles[e+1]=tiles[e]+(rows[e]+127)/128;
    }
    require(offsets.data()==original_rows && tiles.data()==original_tiles);
    const bool small=std::all_of(rows.begin(),rows.end(),[](int m){return m<192;});
    const int first=oracle(rows,n,sw,along,window,workers);
    require(grouped_use_latency_cohort(offsets.data(),int(rows.size()))==small);
    require(grouped_first_wave_panels(order,workers)==first);
    require(explicit_policy.first_consumer_panels(p)==(small?first:-1));
    require(dynamic.first_consumer_panels(p)==(small?first:-1));
  }
}
int main() {
  try {
    // Regression: two first-wave CTAs touch one panel with a one-panel ring,
    // but two panels under the old unwindowed prepare order.
    const int64_t tiles[]={0,2};
    const auto bounded=host_order({1,128,4096},{1,false},tiles,8);
    require(grouped_first_wave_panels(bounded,2)==1);
    auto unbounded=bounded; unbounded.window_m=0;
    require(grouped_first_wave_panels(unbounded,2)==2);
    const int64_t at191[]={0,191},at192[]={0,192};
    const int64_t balanced[]={0,191,382},skewed[]={0,0,382};
    require(grouped_use_latency_cohort(at191,1));
    require(!grouped_use_latency_cohort(at192,1));
    require(grouped_use_latency_cohort(balanced,2));
    require(!grouped_use_latency_cohort(skewed,2));
    std::mt19937 rng(20260919);
    std::vector<std::vector<int>> cases{{},{0},{0,0,0},{191},{192},
        {191,191},{0,382},{127,128,129},{4096,0,1,191,192}};
    for(int trial=0;trial<64;++trial) {
      std::vector<int> rows(1+rng()%12);
      for(int& m:rows) m=rng()%513;
      cases.push_back(rows);
    }
    for(auto rows:cases) for(int mutation=0;mutation<3;++mutation) {
      if(mutation==1) { rows.insert(rows.begin(),0); rows.push_back(0);
        rows.insert(rows.begin()+rows.size()/2,0); }
      if(mutation==2) std::shuffle(rows.begin(),rows.end(),rng);
      for(int n:{1,3,8,33}) for(int sw:{1,2,4,8}) for(bool along:{false,true})
        for(int buffer:{0,128,256,384,4096,8192}) for(int workers:{1,2,17,128,148})
          check_replays(rows,n,sw,along,buffer,workers);
    }
  } catch(const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-grouped-prepare-') as tmp:
            binary = str(Path(tmp)/'prepare')
            result = subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra',
                                     '-Werror','-x','c++','-','-o',binary],
                                    input=source,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result = subprocess.run([binary],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_input_model_matches_independent_worker_recurrence(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        # Compile the actual model/decoder. The oracle below walks EVERY tile
        # of every worker; production uses only each panel's first consumer.
        files = ('producer_consumer.cuh', 'performance_model.cuh')
        body = '\n'.join('\n'.join(line for line in
            (ROOT/'csrc/operators/sm103/detail/grouped'/name).read_text().splitlines()
            if not line.startswith(('#include', '#pragma once'))) for name in files)
        source = r'''
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>
#define CUTLASS_HOST_DEVICE
''' + body + r'''
bool check_dispatch(const std::vector<int>& rows,int n,int sw,bool along,int k,
    int comm,int compute,const fuse::detail::GroupedDispatchServices& service) {
  using namespace fuse::detail;
  std::vector<int64_t> offsets(1,0),tiles(1,0);
  std::vector<int> expert,valid;
  bool small=true;
  for(int e=0;e<int(rows.size());++e) {
    offsets.push_back(offsets.back()+rows[e]);
    tiles.push_back(tiles.back()+(rows[e]+127)/128);
    if(rows[e]>=192) small=false;
    for(int start=0;start<rows[e];start+=128) {
      expert.push_back(e); valid.push_back(std::min(128,rows[e]-start));
    }
  }
  const int panels=int(valid.size());
  GroupedTileOrder order{tiles.data(),int(rows.size()),n,sw,along};
  std::vector<int> touched(panels);
  for(int64_t q=0;q<std::min<int64_t>(compute,order.tiles());++q) {
    const auto tile=order.decode(q);
    touched[tiles[tile.expert]+tile.m]=1;
  }
  const int first=small?int(std::count(touched.begin(),touched.end(),1)):-1;
  // Split selection is separately tested. The queue/row/release oracle below
  // shares only that policy contract, not the scorer's producer arithmetic.
  const int splits=grouped_dispatch_splits(panels,comm,first,int64_t(256)*k);
  std::vector<double> ready(panels);
  for(int cta=0;cta<comm;++cta) {
    double finish=0;
    for(int work=cta;work<panels*splits;work+=comm) {
      const int panel=work/splits,stripe=work%splits;
      const int producers=std::min(splits,(valid[panel]+7)/8);
      if(stripe>=producers) continue;
      int owned=0;
      for(int warp=0;warp<8;++warp)
        for(int row=stripe*8+warp;row<valid[panel];row+=8*producers) ++owned;
      const bool staged=rows[expert[panel]]>128 && k<=12288;
      const double row_us=!staged?service.vector_row_us:
          (splits==1?service.staged_row_us:service.staged_stripe_row_us);
      finish+=owned*row_us;
      ready[panel]=std::max(ready[panel],finish);
    }
  }
  GroupedInputModelResult expected;
  expected.valid=true;
  expected.tiles=int64_t(panels)*n;
  expected.waves=(expected.tiles+compute-1)/compute;
  expected.compute_us=expected.waves*service.tile_us;
  expected.finish_us=expected.compute_us;
  if(panels) {
    expected.copy_us=*std::max_element(ready.begin(),ready.end());
    expected.first_ready_us=*std::min_element(ready.begin(),ready.end());
  }
  for(int worker=0;worker<compute;++worker) {
    double finish=0;
    int64_t critical=-1;
    for(int64_t q=worker;q<expected.tiles;q+=compute) {
      const auto tile=order.decode(q);
      const int64_t panel=tiles[tile.expert]+tile.m;
      if(ready[panel]>finish) { finish=ready[panel]; critical=panel; }
      else if(ready[panel]==finish && critical>=0) critical=std::min(critical,panel);
      finish+=service.tile_us;
    }
    if(finish>expected.finish_us) {
      expected.finish_us=finish; expected.critical_panel=critical;
    } else if(finish==expected.finish_us && expected.critical_panel>=0 && critical>=0)
      expected.critical_panel=std::min(expected.critical_panel,critical);
  }
  expected.exposed_feed_us=expected.finish_us-expected.compute_us;
  const auto actual=score_grouped_dispatch_candidate(order,offsets.data(),k,comm,compute,service);
  if(actual.valid!=expected.valid || actual.tiles!=expected.tiles ||
      actual.waves!=expected.waves || actual.critical_panel!=expected.critical_panel ||
      actual.compute_us!=expected.compute_us || actual.copy_us!=expected.copy_us ||
      actual.first_ready_us!=expected.first_ready_us || actual.finish_us!=expected.finish_us ||
      actual.exposed_feed_us!=expected.exposed_feed_us) {
    fprintf(stderr,"dispatch E=%zu Ntiles=%d K=%d C=%d P=%d sw=%d along=%d splits=%d "
        "expected(F,R,first,critical)=(%f,%f,%f,%lld) got=(%f,%f,%f,%lld)\n",
        rows.size(),n,k,comm,compute,sw,along,splits,
        expected.finish_us,expected.copy_us,expected.first_ready_us,(long long)expected.critical_panel,
        actual.finish_us,actual.copy_us,actual.first_ready_us,(long long)actual.critical_panel);
    return false;
  }
  for(int window:{-1,1,2}) {
    order.window_m=window;
    if(score_grouped_dispatch_candidate(order,offsets.data(),k,comm,compute,service).valid)
      return false;
  }
  return true;
}
int main() {
  std::mt19937 rng(20260919);
  for(int trial=0;trial<2000;++trial) {
    const int experts=1+rng()%12,n=1+rng()%65,workers=1+rng()%148;
    std::vector<int64_t> offsets(1,0);
    for(int e=0;e<experts;++e) offsets.push_back(offsets.back()+rng()%18);
    std::vector<double> ready(offsets.back());
    for(auto& r:ready) r=(rng()%1000)*0.125; // Deliberately OUT of release order.
    const double tile=0.25+(rng()%128)*0.125;
    fuse::detail::GroupedTileOrder order{offsets.data(),experts,n,1<<int(rng()%4),bool(rng()%2),int(rng()%5)};
    const auto score=fuse::detail::score_grouped_input_schedule(order,ready.data(),workers,tile);
    double expected=0;
    for(int worker=0;worker<workers;++worker) {
      double finish=0;
      for(int64_t q=worker;q<order.tiles();q+=workers) {
        const auto t=order.decode(q);
        const auto panel=order.input_ready_index(t.expert,t.m);
        finish=std::max(finish,ready[panel])+tile;
      }
      expected=std::max(expected,finish);
    }
    if(!score.valid || std::abs(expected-score.finish_us)>1e-9 ||
        score.waves!=(order.tiles()+workers-1)/workers || score.exposed_feed_us<0) {
      fprintf(stderr,"trial=%d E=%d Ntiles=%d C=%d along=%d sw=%d window=%d expected=%.9f got=%.9f\n",
          trial,experts,n,workers,order.along_n,order.swizzle,order.window_m,expected,score.finish_us);
      return 1;
    }
    // Translating ALL releases translates the endpoint, not wave service.
    for(auto& r:ready) r+=7;
    const auto shifted=fuse::detail::score_grouped_input_schedule(order,ready.data(),workers,tile);
    if(!ready.empty() && (std::abs(shifted.finish_us-score.finish_us-7)>1e-9 ||
        shifted.compute_us!=score.compute_us)) return 2;
    std::fill(ready.begin(),ready.end(),0);
    const auto prepared=fuse::detail::score_grouped_input_schedule(order,ready.data(),workers,tile);
    if(prepared.finish_us!=prepared.compute_us || prepared.exposed_feed_us!=0) return 3;
  }
  int64_t offsets[]={0,1}; double ready[]={1};
  fuse::detail::GroupedTileOrder order{offsets,1,1,1,false};
  if(fuse::detail::score_grouped_input_schedule(order,ready,0,1).valid ||
     fuse::detail::score_grouped_input_schedule(order,ready,1,0).valid) return 4;
  for(double invalid:{-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    ready[0]=invalid;
    if(fuse::detail::score_grouped_input_schedule(order,ready,1,1).valid) return 5;
    if(fuse::detail::score_grouped_input_schedule(order,ready,1,invalid).valid) return 6;
  }
  const std::vector<fuse::detail::GroupedDispatchServices> services{
      {0.25,0.125,0.0625,0.1875},{0.5,0.0625,0.25,0.125},
      {8,0.25,0.375,0.0625},{0.125,0.375,0.0625,0.25}};
  const int ks[]={256,1024,5120,12280,12288,12296,16384};
  const std::vector<std::vector<int>> fixed_rows{{0},{0,0,0},{191},{192},{191,191},
      {0,382},{127,128,129},{1,7,8,9,15,16,17},{4096,0,1,191,192}};
  for(const auto& rows:fixed_rows) for(int comm:{1,8,20,64})
    for(int compute:{1,3,64}) for(int n:{1,3,33}) for(int sw:{1,2,4,8})
      for(bool along:{false,true}) for(int k:ks) for(const auto& service:services)
        if(!check_dispatch(rows,n,sw,along,k,comm,compute,service)) return 7;
  for(int trial=0;trial<1200;++trial) {
    std::vector<int> rows(1+rng()%12);
    for(int& m:rows) m=trial%4==0?0:int(rng()%(trial%4==1?192:513));
    if(trial%4==3) rows[rng()%rows.size()]=4096;
    const int comm=1+rng()%147,compute=1+rng()%(148-comm);
    if(!check_dispatch(rows,1+rng()%33,1<<int(rng()%4),bool(rng()%2),
        ks[rng()%7],comm,compute,services[rng()%services.size()])) return 8;
  }
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-grouped-model-') as tmp:
            binary = str(Path(tmp)/'model')
            result = subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Werror',
                                     '-x','c++','-','-o',binary],input=source,
                                    capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result = subprocess.run([binary],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_generated_routes_and_production_schedule_properties(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        validation = (ROOT/'benchmarks/sm103/grouped_validation.cuh').read_text()
        generator = validation[validation.index('inline std::vector'):validation.index('\nstruct Peers')]
        schedule = (ROOT/'csrc/operators/sm103/detail/grouped/producer_consumer.cuh').read_text()
        schedule = '\n'.join(line for line in schedule.splitlines()
                             if not line.startswith(('#include', '#pragma once')))
        source = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#define CUTLASS_HOST_DEVICE
namespace fuse { struct GroupedTokenSource { int32_t rank,token,slot; }; }
''' + generator + schedule + r'''
void require(bool value) { if(!value) throw std::runtime_error("property failed"); }
int main() {
  // A fixed seed reproduces the generator; generated counts/configurations
  // exercise the actual C++ schedule, not a Python copy of its arithmetic.
  std::mt19937 choices(20260919);
  for(uint32_t seed=0;seed<256;++seed) {
    const int world=seed%2?8:4, experts=1+choices()%9, tokens=1+choices()%258;
    const int top=1+choices()%std::min(8,world*experts);
    for(int replay=0;replay<16;replay+=2) {
      try {
        const auto a=property_routes(world,experts,tokens,top,seed,replay);
        const auto b=property_routes(world,experts,tokens,top,seed,replay+1);
        std::vector<int> branches(world*tokens*top), expert_tokens(world*tokens);
        for(int e=0;e<world*experts;++e) {
          require(a[e].size()==b[e].size() && a[e].size()<=size_t(world*tokens));
          std::fill(expert_tokens.begin(),expert_tokens.end(),0);
          for(size_t i=0;i<a[e].size();++i) {
            const auto x=a[e][i],y=b[e][a[e].size()-1-i];
            require(x.rank>=0 && x.rank<world && x.token>=0 && x.token<tokens && x.slot>=0 && x.slot<top);
            require(x.rank==y.rank && x.token==y.token && x.slot==y.slot);
            require(++branches[(x.rank*tokens+x.token)*top+x.slot]==1);
            require(++expert_tokens[x.rank*tokens+x.token]==1);
          }
        }
        for(int token=0;token<world*tokens;++token) {
          int count=0; for(int slot=0;slot<top;++slot) count+=branches[token*top+slot];
          require(count==0 || count==top);
          if((replay/2)%4==0) require(count==top);
          if((replay/2)%4==2) require(count==0);
        }
        for(int rank=0;rank<world;++rank) {
          std::vector<int64_t> offsets(1,0);
          std::vector<int64_t> row_offsets(1,0);
          for(int e=0;e<experts;++e) {
            const int64_t rows=a[rank*experts+e].size();
            offsets.push_back(offsets.back()+(rows+127)/128);
            row_offsets.push_back(row_offsets.back()+rows);
          }
          const auto groups=fuse::detail::grouped_dispatch_producer_groups(
              row_offsets.data(),experts);
          int64_t group_oracle=0;
          for(int e=0;e<experts;++e)
            for(int64_t begin=row_offsets[e];begin<row_offsets[e+1];begin+=128)
              group_oracle+=(std::min<int64_t>(128,row_offsets[e+1]-begin)+7)/8;
          require(groups==group_oracle);
          const int n=1+choices()%33,sw=1<<(choices()%4),workers=1+choices()%148;
          fuse::detail::GroupedTileOrder order{offsets.data(),experts,n,sw,bool(choices()%2),int(choices()%5)};
          for(int comm:{1,8,20,40}) for(int compute:{1,64,108,128}) {
            const int effective=fuse::detail::grouped_effective_dispatch_ctas(
                order.row_tiles(),n,comm,compute,groups);
            const int64_t active=std::min<int64_t>(order.tiles(),compute);
            require(effective>=comm && effective<=comm+compute-active);
            if(order.tiles()>=compute || groups<=comm) require(effective==comm);
            if(groups>comm) require(effective<=groups);
            int prior=comm;
            for(int64_t available=0;available<=groups+8;++available) {
              const int current=fuse::detail::grouped_effective_dispatch_ctas(
                  order.row_tiles(),n,comm,compute,available);
              require(current>=prior); prior=current;
            }
          }
          std::vector<int> seen(order.tiles());
          std::vector<int64_t> first(order.row_tiles(),order.tiles());
          for(int worker=0;worker<workers;++worker)
            for(int64_t q=worker;q<order.tiles();q+=workers) {
              const auto t=order.decode(q);
              require(t.valid && t.expert>=0 && t.expert<experts && t.m>=0 && t.n>=0 && t.n<n);
              const auto panel=order.input_ready_index(t.expert,t.m);
              require(panel>=offsets[t.expert] && panel<offsets[t.expert+1]);
              require(order.linear(t.expert,t.m,t.n)==q);
              require(++seen[panel*n+t.n]==1);
              first[panel]=std::min(first[panel],q);
              if(order.window_m && t.m>=order.window_m)
                for(int old_n=0;old_n<n;++old_n)
                  require(order.linear(t.expert,t.m-order.window_m,old_n)<q);
            }
          require(std::all_of(seen.begin(),seen.end(),[](int x){return x==1;}));
          require(std::is_sorted(first.begin(),first.end()));
          std::vector<int> first_wave(order.row_tiles());
          for(int64_t q=0;q<std::min<int64_t>(workers,order.tiles());++q) {
            const auto t=order.decode(q);
            first_wave[order.input_ready_index(t.expert,t.m)]=1;
          }
          require(fuse::detail::grouped_first_wave_panels(order,workers)==
              std::count(first_wave.begin(),first_wave.end(),1));
        }
      } catch(const std::exception& e) {
        fprintf(stderr,"seed=%u replay=%d world=%d experts=%d tokens=%d top=%d: %s\n",seed,replay,world,experts,tokens,top,e.what());
        return 1;
      }
    }
  }
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-grouped-properties-') as tmp:
            binary = str(Path(tmp)/'properties')
            result = subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Werror',
                                     '-x','c++','-','-o',binary],input=source,
                                    capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result = subprocess.run([binary],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_swapab_keeps_physical_output_and_logical_ready_units(self):
        # D^T column-major aliases D row-major, including a partial last tile.
        # This is a layout/descriptor test; CUDA replay validates computation.
        for m in (1,15,16,17,127,128,129,192,255,256,257):
            for n in (128,256,384):
                for row in range(m):
                    for col in (0,n//2,n-1):
                        self.assertEqual(row*n+col,col+row*n)
                widths=[min(128,m-start) for start in range(0,m,128)]
                rounded=[(x+15)//16*16 for x in widths]
                self.assertTrue(all(16<=x<=128 for x in rounded))
                self.assertLessEqual(sum(rounded)-m,15)
        api=(ROOT/'csrc/operators/sm103/api/grouped.cuh').read_text()
        self.assertIn('std::swap(g.mainloop.ptr_A,g.mainloop.ptr_B)',api)
        self.assertIn('GroupedDispatchComm<>, Selector>>',api)
        self.assertIn('prepare_grouped_invocation<TileM,SwapAB>',api)
        scheduler=(ROOT/'csrc/operators/sm103/detail/grouped/persistent_gemm.cuh').read_text()
        self.assertIn('const int physical_m=t.m*SmMode+cluster_rank_;',scheduler)
        self.assertIn('SwapAB ? t.n : physical_m, SwapAB ? physical_m : t.n',scheduler)
        header=(ROOT/'include/fuse/operators/primitives/grouped_gemm.h').read_text()
        self.assertIn('bool swap_ab = false;',header)

    def test_dispatch_row_address_prefetch_matches_copy_ownership(self):
        body=(ROOT/'csrc/operators/sm103/detail/grouped/a2a_gemm.cuh').read_text()
        bench=(ROOT/'benchmarks/sm103/grouped_bf16.cu').read_text()
        self.assertEqual(bench.count('q.dispatch_copy=policy.dispatch_copy;'),2)
        for valid in range(1,129):
            for slots in (1,2,3,4,8):
                for batch in (1,2,4,8,16,32,64,128):
                    all_rows=[]
                    for warp in range(slots):
                        expected=[r for begin in range(warp*batch,valid,slots*batch)
                                  for r in range(begin,min(valid,begin+batch))]
                        actual=[]
                        for lane in range(32):
                            owned=lane
                            while True:
                                row=(owned//batch)*slots*batch+warp*batch+owned%batch
                                if row>=valid: break
                                actual.append(row); owned+=32
                        self.assertEqual(sorted(actual),expected)
                        all_rows.extend(actual)
                    self.assertEqual(sorted(all_rows),list(range(valid)))
        self.assertIn('const Bf16* src=row_sources[begin+r];',body)
        self.assertIn('SM80_CP_ASYNC_CACHEGLOBAL<uint4>::copy',body)
        self.assertIn('if(lane==0) cute::tma_store_wait<0>();',body)
        self.assertIn('if (args.params.cp_async_g2s)',body)
        self.assertIn('copy<SharedPanels,true>',body)
        self.assertIn('copy<SharedPanels,false>',body)
        # The invocation-static choice is outside copy(), hence outside its
        # panel loop; neither generated hot path contains a method branch.
        selector=body[body.index('void copy_selected'):body.index('void finalize')]
        copier=body[body.index('void copy(const Params&'):body.index('void copy_selected')]
        self.assertIn('args.params.cp_async_g2s',selector)
        self.assertNotIn('args.params.cp_async_g2s',copier)

    def test_dispatch_staged_rows_are_contiguous_and_covered(self):
        body=(ROOT/'csrc/operators/sm103/detail/grouped/a2a_gemm.cuh').read_text()
        stage_bytes=int(re.search(r'kStageBytes = (\d+) \* 1024',body)[1])*1024
        for k in (8,24,256,1408,2048,3072,4096,5120,6144,7168,32768):
            if k*2>stage_bytes:
                self.assertIn('int64_t(p.columns)*sizeof(Bf16)<=kStageBytes',body)
                continue  # A whole row cannot fit: production uses vector copy.
            batch_rows=1
            while batch_rows<128 and batch_rows*2*k*2<=stage_bytes:
                batch_rows*=2
            self.assertLessEqual(batch_rows*k*2,stage_bytes)
            for valid in (1,7,63,127,128):
                rows=[r for begin in range(0,valid,batch_rows)
                      for r in range(begin,min(valid,begin+batch_rows))]
                self.assertEqual(rows,list(range(valid)))
                for slots in (1,2,3,4,8):
                    selected=batch_rows
                    while selected>1 and (valid+selected-1)//selected<slots:
                        selected//=2
                    copied=[r for slot in range(slots)
                            for begin in range(slot*selected,valid,slots*selected)
                            for r in range(begin,min(valid,begin+selected))]
                    self.assertEqual(sorted(copied),list(range(valid)))
        for valid in (1,7,9,63,127,128):
            for producers in (1,2,4,8,16):
                copied=[r for stripe in range(producers)
                        for start in range(stripe*8,valid,8*producers)
                        for r in range(start,min(valid,start+8))]
                self.assertEqual(sorted(copied),list(range(valid)))
        self.assertIn('rows>TileM',body)
        self.assertIn('offset+=warps*producers',body)
        self.assertIn('mbarrier.inval.shared::cta.b64',body)
        self.assertIn('tma_store_wait_all();',body)

    def test_dispatch_vector_batches_cover_tail_once(self):
        body=(ROOT/'csrc/operators/sm103/detail/grouped/a2a_gemm.cuh').read_text()
        depth=int(re.search(r'kVectorsPerThread = (\d+)',body)[1])
        for k in (8,24,256,1024,1408,2048,3072,4096,5120,6144,7168):
            seen=[]
            for lane in range(32):
                for col in range(lane*8,k,depth*32*8):
                    for i in range(depth):
                        start=col+i*32*8
                        if start<k: seen.extend(range(start,start+8))
            self.assertEqual(sorted(seen),list(range(k)))
        self.assertIn('col += kVectorsPerThread * 32 * 8',body)
        self.assertIn('uint4 values[kVectorsPerThread]',body)

    def test_measurement_accepts_first_stable_not_fastest(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        body = (ROOT / 'benchmarks/sm103/grouped_measurement.cuh').read_text()
        types = body[body.index('inline double percentile'):body.index('inline Samples measure')]
        formal = body[body.index('  Timer timer(ops);'):body.index('\n}  // namespace grouped_measurement')]
        source = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
bool all_bad=false;
int calls=0;
struct Timer {
  explicit Timer(int) { calls=0; }
  std::vector<float> sample() {
    const int round=calls/60, index=calls++%60;
    const bool bad=all_bad || round==0;
    const float ms=bad ? (index<35?10.f:20.f) : (round==1?30.f:1.f);
    return {ms*.5f,ms};
  }
};
''' + types + '\nSamples exercise() { int ops=0; Samples out; out.warmup=128;\n' + formal + r'''
int main() {
  auto result=exercise();
  assert(result.rounds.size()==2 && calls==120);
  assert(result.p50==30 && result.drift==0 && result.warmup==148);
  assert(result.rounds[0].drift>.05 && result.rounds[0].ranks_ms.size()==50);
  all_bad=true; result=exercise();
  assert(result.rounds.size()==3 && calls==180 && result.drift>.05);
  assert(result.warmup==158);
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-grouped-timer-') as tmp:
            binary = str(Path(tmp) / 'timer')
            compiled = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall', '-Werror',
                                       '-x', 'c++', '-', '-o', binary],
                                      input=source, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_private_implementation_layers(self):
        root = ROOT / 'csrc/operators/sm103/detail/grouped'
        for header in root.glob('*.cuh'):
            for dependency in re.findall(r'^#include "([^"]+)"', header.read_text(), re.M):
                if dependency.startswith('fuse/'):
                    self.assertTrue(dependency.startswith(('fuse/arch/', 'fuse/profiling/')) or
                                    dependency == 'fuse/operators/primitives/grouped_gemm.h')
                else:
                    self.assertNotIn('/', dependency, (header.name, dependency))
                    self.assertTrue((root / dependency).is_file(), dependency)
        for filename, role in [('a2a_gemm.cuh', 'GroupedDispatchComm'),
                               ('gemm_a2a.cuh', 'GroupedCombineComm')]:
            body = (root / filename).read_text()
            self.assertIn(role, body)
            self.assertIn('GroupedMonolithicGemm<', body)
            self.assertNotIn('GroupedTokenComm', body)

    def test_tile_order_and_ctasp_coverage(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        body = (ROOT / 'csrc/operators/sm103/detail/grouped/producer_consumer.cuh').read_text()
        body = '\n'.join(line for line in body.splitlines()
                         if not line.startswith(('#include', '#pragma once')))
        source = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
#define CUTLASS_HOST_DEVICE
''' + body + r'''
using Order = fuse::detail::GroupedTileOrder;

void check(const std::vector<int>& rows, int n, int sw, bool along_n, int window) {
  std::vector<int64_t> offsets(1, 0);
  for (int m : rows) offsets.push_back(offsets.back() + (int64_t(m) + 127) / 128);
  Order order{offsets.data(), int(rows.size()), n, sw, along_n, window};
  std::vector<Order::Tile> expected;
  // Independent nested-loop oracle, not a copy of division-based decode.
  for (int e = 0; e < int(rows.size()); ++e) {
    const int total_m = int(offsets[e+1] - offsets[e]);
    const int step=window?window:std::max(1,total_m);
    for(int first_m=0;first_m<total_m;first_m+=step) {
    const int mt = std::min(step,total_m-first_m);
    const int outer = along_n ? mt : n, middle = along_n ? n : mt;
    for (int band = 0; band < outer; band += sw)
      for (int major = 0; major < middle; ++major)
        for (int minor = band; minor < std::min(band + sw, outer); ++minor)
          expected.push_back({e, first_m+(along_n ? minor : major), along_n ? major : minor, true});
    }
  }
  assert(int64_t(expected.size()) == order.tiles());
  std::vector<int> seen(order.tiles());
  std::vector<int64_t> first(order.row_tiles(), order.tiles());
  for (int64_t q = 0; q < order.tiles(); ++q) {
    const auto t = order.decode(q), r = expected[q];
    assert(t.valid && t.expert == r.expert && t.m == r.m && t.n == r.n);
    assert(order.linear(t.expert, t.m, t.n) == q);
    const auto a = order.input_ready_index(t.expert, t.m);
    const auto d = order.output_ready_index(t.expert, t.m, t.n);
    assert(a >= 0 && a < order.row_tiles() && d >= 0 && d < order.tiles());
    assert(++seen[d] == 1);
    first[a] = std::min(first[a], q);
  }
  assert(std::is_sorted(first.begin(), first.end()));
  // Every reader of an old slot precedes every reader of its replacement.
  // A static CTA queue cannot wait for a new window while retaining an
  // unexecuted old-window task behind it, for either raster or partial bands.
  if(window) for(int e=0;e<int(rows.size());++e)
    for(int m=window;m<offsets[e+1]-offsets[e];++m)
      for(int old_n=0;old_n<n;++old_n) for(int new_n=0;new_n<n;++new_n)
        assert(order.linear(e,m-window,old_n)<order.linear(e,m,new_n));
  assert(!order.decode(-1).valid && !order.decode(order.tiles()).valid);
  for (int comm : {1, 8, 20}) for (int compute : {1, 3, 8, 64, 128}) {
    std::fill(seen.begin(), seen.end(), 0);
    // Physical CTA prefix is COMM, stride is only COMPUTE, including tails.
    for (int block = comm; block < comm + compute; ++block)
      for (int64_t q = block - comm; q < order.tiles(); q += compute)
        assert(++seen[q] == 1);
    for (int count : seen) assert(count == 1);
  }
}

int main() {
  { int64_t small[]={0,0,191,382},large[]={0,64,256,320};
    assert(fuse::detail::grouped_use_latency_cohort(small,3));
    assert(!fuse::detail::grouped_use_latency_cohort(large,3)); }
  { int64_t rows[]={0,1,9,25,153};
    assert(fuse::detail::grouped_dispatch_producer_groups(rows,4)==20); }
  assert(fuse::detail::grouped_effective_dispatch_ctas(8,12,20,128,64)==52);
  assert(fuse::detail::grouped_effective_dispatch_ctas(8,12,20,128,32)==32);
  assert(fuse::detail::grouped_effective_dispatch_ctas(8,12,20,128,8)==20);
  assert(fuse::detail::grouped_effective_dispatch_ctas(1,112,20,128,32)==32);
  assert(fuse::detail::grouped_effective_dispatch_ctas(16,24,20,128,256)==20);
  assert(fuse::detail::grouped_effective_dispatch_ctas(0,24,20,128,256)==148);
  // New tail sharing: each row is copied once; every nonempty stripe arrives
  // once. Empty stripes must not increase the final publisher's target.
  for(int panels:{21,36,64,72,108,144,257}) for(int c:{8,20,40,64}) {
    const int begin=panels-panels%c;
    if(!begin || begin==panels) continue;
    for(int rows:{1,7,17,64,127,128}) {
      const int producers=(rows+7)/8;
      std::vector<int> seen((panels-begin)*rows),arrivals(panels-begin);
      for(int worker=0;worker<c;++worker)
        for(int work=worker;work<(panels-begin)*16;work+=c) {
          const int p=work/16,stripe=work%16;
          if(stripe>=producers) continue;
          ++arrivals[p];
          for(int warp=0;warp<8;++warp)
            for(int row=stripe*8+warp;row<rows;row+=8*producers) ++seen[p*rows+row];
        }
      for(int x:seen) assert(x==1);
      for(int x:arrivals) assert(x==producers);
    }
  }
  // Cooperative Dispatch still copies every row exactly once and requires only
  // one consumer ready per panel, including tails and fewer rows than warps.
  for (int panels : {0,1,2,3,8,17,32,128}) for (int c : {1,8,20,40,64,128}) {
    int splits=fuse::detail::grouped_dispatch_splits(panels,c,-1);
    assert(splits>=1 && splits<=16);
    if(panels>=c || panels==0) assert(splits==1);
    else assert(splits==std::min(16,c/panels));
    if(panels>0 && panels<c) {
      int small=c/panels;
      if(small==1 && c%panels) small=2;
      assert(fuse::detail::grouped_dispatch_splits(panels,c,1)==std::min(16,small));
      assert(fuse::detail::grouped_dispatch_splits(panels,c,-1)==splits);
    }
    if(panels>=c && panels>6 && c>6) assert(fuse::detail::grouped_dispatch_splits(panels,c,6)==
        std::min(16,c/6));
    if(panels>=32 && c==20) {
      assert(fuse::detail::grouped_dispatch_splits(panels,c,6,512*1024)==2);
      assert(fuse::detail::grouped_dispatch_splits(panels,c,6,1024*1024)==3);
    }
    for(int rows : {1,7,8,9,16,31,64,127,128}) {
      const int producers=std::min(splits,(rows+7)/8);
      std::vector<int> seen(rows);
      for(int stripe=0;stripe<splits;++stripe) {
        if(stripe>=producers) continue;
        for(int warp=0;warp<8;++warp)
          for(int row=stripe*8+warp;row<rows;row+=8*producers) ++seen[row];
      }
      for(int visits:seen) assert(visits==1);
    }
  }
  for (const auto& rows : std::vector<std::vector<int>>{
      {}, {0}, {0,0,0}, {1}, {1,8,0,129,256,0}, {8192,1,0,17,127,128,129},
      {3,1001,12,23,384,999,0,0,0,4097}})
    for (int n : {1,3,8,11,32}) for (int sw : {1,2,4,8})
      for (bool along_n : {false,true}) for(int window : {0,1,2,3,8}) check(rows, n, sw, along_n, window);
  // Dynamic replay changes the same offsets storage; no cached host tile count.
  std::vector<int64_t> offsets{0,0,2,2};
  Order dynamic{offsets.data(), 3, 7, 4, true};
  assert(dynamic.decode(0).expert == 1 && dynamic.tiles() == 14);
  offsets = {0,3,3,8};
  dynamic.row_tile_offsets = offsets.data();
  assert(dynamic.decode(0).expert == 0 && dynamic.decode(21).expert == 2);
  assert(dynamic.tiles() == 56);
  // Tile totals and ready indexing must not silently overflow int32.
  int64_t large[] = {0, 1LL<<25, 1LL<<26};
  Order wide{large, 2, 256, 8, false};
  for (int64_t q : {0LL, (1LL<<33)-1, 1LL<<33, (1LL<<34)-1}) {
    const auto t = wide.decode(q);
    assert(t.valid && wide.linear(t.expert,t.m,t.n) == q);
  }
  std::cout << "grouped order: coverage, tails, empty experts, first-use, replay, int64 passed\n";
}
'''
        with tempfile.TemporaryDirectory(prefix='fuse-grouped-test-') as tmp:
            binary = str(Path(tmp) / 'grouped_order')
            compiled = subprocess.run([compiler, '-std=c++17', '-O2', '-Wall',
                                       '-Wextra', '-Werror', '-x', 'c++', '-', '-o', binary],
                                      input=source, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()

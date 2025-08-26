#include <vector>
#include <cstdio>
#include kklm.h
int main(){
  std::vector<float> x(16); for(size_t i=0;i<x.size();++i) x[i] = (float(i)-8.f)*0.25f;
  auto qp = kllm::choose_symmetric_int8_scale(x.data(), x.size());
  std::printf(scale=%gn, qp.scale);
  return 0;
}

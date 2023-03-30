#include <TorchUtils.h>

using namespace std;
using namespace ttk;

#ifdef TTK_ENABLE_TORCH
void TorchUtils::copyTensor(torch::Tensor &a, torch::Tensor &b) {
  b = a.detach().clone();
  b.requires_grad_(a.requires_grad());
}
#endif

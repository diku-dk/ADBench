#include "../../shared/gmm.h"
#include "../../shared/ITest.h"

#include "../../shared/GMMData.h"

extern "C" {
#include "gmm.h"
}

#include <vector>

#undef NDEBUG
#define USE_FWD
#include <assert.h>
#include <iostream>

class FutharkGMM : public ITest<GMMInput, GMMOutput> {
private:
  GMMInput input;
  GMMOutput result;

  struct futhark_context_config *cfg;
  struct futhark_context *ctx;
  struct futhark_f64_1d *alphas;
  struct futhark_f64_2d *means;
  struct futhark_f64_2d *icf;
  struct futhark_f64_2d *x;
  double w_gamma;
  int64_t w_m;

public:
  // This function must be called before any other function.
  virtual void prepare(GMMInput&& input) override;

  virtual void calculate_objective(int times) override;
  virtual void calculate_jacobian(int times) override;
  virtual GMMOutput output() override;

  virtual ~FutharkGMM();
};


// This function must be called before any other function.
void FutharkGMM::prepare(GMMInput&& input)
{

  this->input = input;
  int Jcols = (this->input.k * (this->input.d + 1) * (this->input.d + 2)) / 2;
  result = { 0, std::vector<double>(Jcols) };

  this->cfg = futhark_context_config_new();
  this->ctx = futhark_context_new(this->cfg);

  futhark_context_config_set_debugging(this->cfg, true);
  futhark_context_config_set_logging(this->cfg, true);

  assert((this->alphas = futhark_new_f64_1d(this->ctx, input.alphas.data(), input.k)) != nullptr);
  assert((this->means = futhark_new_f64_2d(this->ctx, input.means.data(), input.k, input.d)) != nullptr);
  assert((this->icf = futhark_new_f64_2d(this->ctx, input.icf.data(), input.k, (input.d * (input.d + 1))/ 2)) != nullptr);
  assert((this->x = futhark_new_f64_2d(this->ctx, input.x.data(), input.n, input.d)) != nullptr);
  this->w_gamma = input.wishart.gamma;
  this->w_m = input.wishart.m;
}

FutharkGMM::~FutharkGMM()
{
  assert(futhark_free_f64_1d(this->ctx, this->alphas) == 0);
  assert(futhark_free_f64_2d(this->ctx, this->means) == 0);
  assert(futhark_free_f64_2d(this->ctx, this->icf) == 0);
  assert(futhark_free_f64_2d(this->ctx, this->x) == 0);

  futhark_context_free(this->ctx);
  futhark_context_config_free(this->cfg);
}

GMMOutput FutharkGMM::output()
{
  return result;
}

void FutharkGMM::calculate_objective(int times)
{
  for (int i = 0; i < times; i++) {
    assert(futhark_entry_calculate_objective(
      this->ctx,
      &(this->result.objective),
      this->alphas,
      this->means,
      this->icf,
      this->x,
      this->w_gamma,
      this->w_m
      ) == 0);

    assert(futhark_context_sync(this->ctx) == 0);
  }
}

#ifndef USE_FWD
void FutharkGMM::calculate_jacobian(int times)
{
  struct futhark_f64_1d *J_alphas = nullptr;
  struct futhark_f64_2d *J_means = nullptr;
  struct futhark_f64_2d *J_icf = nullptr;

  double* alphas_gradient_part = result.gradient.data();
  double* means_gradient_part = result.gradient.data() + input.alphas.size();
  double* icf_gradient_part =
    result.gradient.data() +
    input.alphas.size() +
    input.means.size();

  for (int i = 0; i < times; i++) {
    assert(futhark_entry_calculate_jacobian(
      this->ctx,
      &J_alphas,
      &J_means,
      &J_icf,
      this->alphas,
      this->means,
      this->icf,
      this->x,
      this->w_gamma,
      this->w_m
    ) == 0);

    assert(futhark_values_f64_1d(this->ctx, J_alphas, alphas_gradient_part) == 0);
    assert(futhark_values_f64_2d(this->ctx, J_means, means_gradient_part) == 0);
    assert(futhark_values_f64_2d(this->ctx, J_icf, icf_gradient_part) == 0);
    assert(futhark_context_sync(this->ctx) == 0);
  }
}

#else
// Uses forward mode
void FutharkGMM::calculate_jacobian(int times)
{
  struct futhark_f64_1d *fut_grad = nullptr;

  for (int i = 0; i < times; i++)
  {
    assert(futhark_entry_calculate_jacobian_fwd(
      this->ctx,
      &fut_grad,
      this->alphas,
      this->means,
      this->icf,
      this->x,
      this->w_gamma,
      this->w_m
    ) == 0);

    assert(futhark_values_f64_1d(this->ctx, fut_grad, result.gradient.data()) == 0);
    assert(futhark_context_sync(this->ctx) == 0);
  }

}

#endif

extern "C" DLL_PUBLIC ITest<GMMInput, GMMOutput>* get_gmm_test()
{
  return new FutharkGMM();
}

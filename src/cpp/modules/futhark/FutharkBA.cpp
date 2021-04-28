#include "../../shared/ba.h"
#include "../../shared/ITest.h"
#include "../../shared/BAData.h"

extern "C" {
#include "ba.h"
}

#include <vector>

#undef NDEBUG
#include <assert.h>

class FutharkBA : public ITest<BAInput, BAOutput> {
private:
  BAInput input;
  BAOutput result;

  struct futhark_context_config *cfg;
  struct futhark_context *ctx;
  struct futhark_f64_2d *cams;
  struct futhark_f64_2d *X;
  struct futhark_f64_1d *w;
  struct futhark_i32_2d *obs;
  struct futhark_f64_2d *feats;

public:
  // This function must be called before any other function.
  virtual void prepare(BAInput&& input) override;

  virtual void calculate_objective(int times) override;
  virtual void calculate_jacobian(int times) override;
  virtual BAOutput output() override;

  virtual ~FutharkBA();
};


// This function must be called before any other function.
void FutharkBA::prepare(BAInput&& input)
{
  this->input = input;
  result = { std::vector<double>(2 * this->input.p), std::vector<double>(this->input.p), BASparseMat(this->input.n, this->input.m, this->input.p) };
  int n_new_cols = BA_NCAMPARAMS + 3 + 1;

  this->cfg = futhark_context_config_new();
  this->ctx = futhark_context_new(this->cfg);

  assert((this->cams = futhark_new_f64_2d(this->ctx, input.cams.data(), input.n, 11)) != nullptr);
  assert((this->X = futhark_new_f64_2d(this->ctx, input.X.data(), input.m, 3)) != nullptr);
  assert((this->w = futhark_new_f64_1d(this->ctx, input.w.data(), input.p)) != nullptr);
  assert((this->feats = futhark_new_f64_2d(this->ctx, input.feats.data(), input.p, 2)) != nullptr);
  assert((this->obs = futhark_new_i32_2d(this->ctx, input.obs.data(), input.p, 2)) != nullptr);
}

FutharkBA::~FutharkBA()
{
  assert(futhark_free_f64_2d(this->ctx, this->cams) == 0);
  assert(futhark_free_f64_2d(this->ctx, this->X) == 0);
  assert(futhark_free_f64_1d(this->ctx, this->w) == 0);
  assert(futhark_free_f64_2d(this->ctx, this->feats) == 0);
  assert(futhark_free_i32_2d(this->ctx, this->obs) == 0);

  futhark_context_free(this->ctx);
  futhark_context_config_free(this->cfg);
}

BAOutput FutharkBA::output()
{
  return result;
}

void FutharkBA::calculate_objective(int times)
{
  struct futhark_f64_2d *reproj_err = nullptr;
  struct futhark_f64_1d *w_err = nullptr;

  for (int i = 0; i < times; ++i) {
    if (i != 0) {
      futhark_free_f64_2d(this->ctx, reproj_err);
      futhark_free_f64_1d(this->ctx, w_err);
    }

    assert(futhark_entry_calculate_objective
           (this->ctx,
            &reproj_err, &w_err,
            this->cams, this->X, this->w, this->obs, this->feats)
           == 0);
    assert(futhark_context_sync(this->ctx) == 0);
  }

  this->result.reproj_err.resize(this->input.p * 2);
  this->result.w_err.resize(this->input.p);

  assert(futhark_values_f64_2d(this->ctx, reproj_err, this->result.reproj_err.data()) == 0);
  assert(futhark_values_f64_1d(this->ctx, w_err, this->result.w_err.data()) == 0);

  futhark_free_f64_2d(this->ctx, reproj_err);
  futhark_free_f64_1d(this->ctx, w_err);
}

void FutharkBA::calculate_jacobian(int times)
{
  struct futhark_i32_1d *J_rows = nullptr;
  struct futhark_i32_1d *J_cols = nullptr;
  struct futhark_f64_1d *J_vals = nullptr;

  for (int i = 0; i < times; ++i) {
    if (i != 0) {
      futhark_free_i32_1d(this->ctx, J_rows);
      futhark_free_i32_1d(this->ctx, J_cols);
      futhark_free_f64_1d(this->ctx, J_vals);
    }

    assert(futhark_entry_calculate_jacobian
           (this->ctx,
            &J_rows, &J_cols, &J_vals,
            this->cams, this->X, this->w, this->obs, this->feats)
           == 0);
    assert(futhark_context_sync(this->ctx) == 0);
  }

  int rows_len = futhark_shape_i32_1d(this->ctx, J_rows)[0];
  int cols_len = futhark_shape_i32_1d(this->ctx, J_cols)[0];
  this->result.J.rows.resize(rows_len);
  this->result.J.cols.resize(cols_len);
  this->result.J.vals.resize(cols_len);

  assert(futhark_values_i32_1d(this->ctx, J_rows, this->result.J.rows.data()) == 0);
  assert(futhark_values_i32_1d(this->ctx, J_cols, this->result.J.cols.data()) == 0);
  assert(futhark_values_f64_1d(this->ctx, J_vals, this->result.J.vals.data()) == 0);

  futhark_free_i32_1d(this->ctx, J_rows);
  futhark_free_i32_1d(this->ctx, J_cols);
  futhark_free_f64_1d(this->ctx, J_vals);
}

extern "C" DLL_PUBLIC ITest<BAInput, BAOutput>* get_ba_test()
{
  return new FutharkBA();
}

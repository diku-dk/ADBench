#include "../../shared/ITest.h"
#include "../../shared/HandData.h"

#include "FutharkUtil.h"

extern "C" {
#include "omnibus.h"
}

#include <vector>
#include <cstring>
#include <iostream>
#undef NDEBUG
#include <assert.h>

class FutharkHand : public ITest<HandInput, HandOutput> {
private:
  HandInput _input;
  HandOutput _output;
  bool _complicated = false;

  struct futhark_context_config *_cfg;
  struct futhark_context *_ctx;

  struct futhark_i32_1d *_parents;
  struct futhark_f64_3d *_base_relatives;
  struct futhark_f64_3d *_inverse_base_absolutes;
  struct futhark_f64_2d *_weights;
  struct futhark_f64_2d *_base_positions;
  struct futhark_i32_2d *_triangles;
  struct futhark_i32_1d *_correspondences;
  struct futhark_f64_2d *_points;
  struct futhark_f64_1d *_theta;
  struct futhark_f64_1d *_us;

public:
  // This function must be called before any other function.
  virtual void prepare(HandInput&& input) override;

  virtual void calculate_objective(int times) override;
  virtual void calculate_jacobian(int times) override;
  virtual HandOutput output() override;

  virtual ~FutharkHand();
};

struct futhark_f64_2d* pack_light_matrix(struct futhark_context *ctx,
                                         const LightMatrix<double>& m) {
  int nrows = m.nrows_;
  int ncols = m.ncols_;
  int mat_size = nrows*ncols;
  double *data = new double[mat_size];

  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      data[i*ncols+j] = m(i,j);
    }
  }

  struct futhark_f64_2d *fut =
    futhark_new_f64_2d(ctx, data, nrows, ncols);

  assert(fut != nullptr);
  assert(futhark_context_sync(ctx) == 0);

  delete[] data;

  return fut;
}

struct futhark_f64_3d* pack_vector_light_matrix(struct futhark_context *ctx,
                                                const std::vector<LightMatrix<double>>& v) {
  int num_mats = v.size();
  int nrows = v[0].nrows_;
  int ncols = v[0].ncols_;
  int mat_size = nrows*ncols;
  double *data = new double[num_mats * mat_size];

  for (int l = 0; l < num_mats; l++) {
    auto& m = v[l];
    for (int i = 0; i < nrows; i++) {
      for (int j = 0; j < ncols; j++) {
        data[l*mat_size + i*ncols + j] = m(i,j);
      }
    }
  }

  struct futhark_f64_3d *fut =
    futhark_new_f64_3d(ctx, data, num_mats, ncols, nrows);

  assert(fut != nullptr);
  assert(futhark_context_sync(ctx) == 0);

  delete[] data;

  return fut;
}

struct futhark_i32_2d* pack_triangles(struct futhark_context *ctx,
                                      const std::vector<Triangle>& v) {
  int ntriangles = v.size();
  int *data = new int[ntriangles*3];

  for (int i = 0; i < ntriangles; i++) {
    std::memcpy(data + i*3, v[i].verts, 3*sizeof(int));
  }

  struct futhark_i32_2d *fut =
    futhark_new_i32_2d(ctx, data, ntriangles, 3);

  assert(fut != nullptr);
  assert(futhark_context_sync(ctx) == 0);

  delete[] data;

  return fut;
}

// This function must be called before any other function.
void FutharkHand::prepare(HandInput&& input)
{

  _input = input;
  _complicated = _input.us.size() != 0;
  int err_size = 3 * _input.data.correspondences.size();
  int ncols = (_complicated ? 2 : 0) + _input.theta.size();
  _output = { std::vector<double>(err_size), ncols, err_size, std::vector<double>(err_size * ncols) };

  _cfg = futhark_context_config_new();
  _ctx = futhark_context_new(_cfg);

  _parents =
    futhark_new_i32_1d(_ctx, _input.data.model.parents.data(), _input.data.model.parents.size());
  _base_relatives =
    pack_vector_light_matrix(_ctx, _input.data.model.base_relatives);
  _inverse_base_absolutes =
    pack_vector_light_matrix(_ctx, _input.data.model.inverse_base_absolutes);
  _weights =
    pack_light_matrix(_ctx, _input.data.model.weights);
  _base_positions =
    pack_light_matrix(_ctx, _input.data.model.base_positions);
  _triangles =
    pack_triangles(_ctx, _input.data.model.triangles);
  _correspondences =
    futhark_new_i32_1d(_ctx, _input.data.correspondences.data(), _input.data.correspondences.size());
  _points =
    pack_light_matrix(_ctx, _input.data.points);
  _theta =
    futhark_new_f64_1d(_ctx, _input.theta.data(), _input.theta.size());
  _us =
    futhark_new_f64_1d(_ctx, _input.us.data(), _input.us.size());
}

FutharkHand::~FutharkHand()
{
  assert(futhark_free_i32_1d(_ctx, _parents) == 0);
  assert(futhark_free_f64_3d(_ctx, _base_relatives) == 0);
  assert(futhark_free_f64_3d(_ctx, _inverse_base_absolutes) == 0);
  assert(futhark_free_f64_2d(_ctx, _weights) == 0);
  assert(futhark_free_f64_2d(_ctx, _base_positions) == 0);
  assert(futhark_free_i32_2d(_ctx, _triangles) == 0);
  assert(futhark_free_i32_1d(_ctx, _correspondences) == 0);
  assert(futhark_free_f64_2d(_ctx, _points) == 0);
  assert(futhark_free_f64_1d(_ctx, _theta) == 0);
  assert(futhark_free_f64_1d(_ctx, _us) == 0);

  futhark_context_free(_ctx);
  futhark_context_config_free(_cfg);
}

HandOutput FutharkHand::output()
{
  return _output;
}

void FutharkHand::calculate_objective(int times)
{
  struct futhark_f64_2d *out = nullptr;

  for (int i = 0; i < times; i++) {
    if (i != 0) {
      futhark_free_f64_2d(_ctx, out);
    }

    FUTHARK_SUCCEED
      (_ctx,
       futhark_entry_hand_calculate_objective
       (_ctx,
        &out,
        _parents,
        _base_relatives,
        _inverse_base_absolutes,
        _weights,
        _base_positions,
        _triangles,
        _input.data.model.is_mirrored,
        _correspondences,
        _points,
        _theta,
        _us
        ));

    assert(futhark_context_sync(_ctx) == 0);
  }

  futhark_values_f64_2d(_ctx, out, _output.objective.data());
  FUTHARK_SUCCEED(_ctx, futhark_context_sync(_ctx));
  futhark_free_f64_2d(_ctx, out);
}

void FutharkHand::calculate_jacobian(int times)
{
  struct futhark_f64_2d *J = nullptr;

  for (int i = 0; i < times; i++) {
    if (i != 0) {
      futhark_free_f64_2d(_ctx, J);
    }

    FUTHARK_SUCCEED
      (_ctx,
       futhark_entry_hand_calculate_jacobian
       (_ctx,
        &J,
        _parents,
        _base_relatives,
        _inverse_base_absolutes,
        _weights,
        _base_positions,
        _triangles,
        _input.data.model.is_mirrored,
        _correspondences,
        _points,
        _theta,
        _us
        ));

    FUTHARK_SUCCEED(_ctx, futhark_context_sync(_ctx));
  }

  FUTHARK_SUCCEED(_ctx, futhark_values_f64_2d(_ctx, J, _output.jacobian.data()));
  FUTHARK_SUCCEED(_ctx,futhark_context_sync(_ctx));
  FUTHARK_SUCCEED(_ctx, futhark_free_f64_2d(_ctx, J));
}

extern "C" DLL_PUBLIC ITest<HandInput, HandOutput>* get_hand_test()
{
  return new FutharkHand();
}

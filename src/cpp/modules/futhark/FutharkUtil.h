#pragma once

#define FUTHARK_SUCCEED(ctx, e) if (e != 0) { fprintf(stderr, "%s:%d: %s", __func__, __LINE__, futhark_context_get_error(ctx)); abort(); }

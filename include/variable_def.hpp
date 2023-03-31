#if defined(CODI_REVERSE_TYPE)  // reverse mode AD
#include "codi.hpp"
#include "codi/tools/data/externalFunctionUserData.hpp"

#if defined(HAVE_OMP)
using mlpdouble = codi::RealReverseIndexOpenMP;
#else
#if defined(CODI_INDEX_TAPE)
using mlpdouble = codi::RealReverseIndex;
#else
using mlpdouble = codi::RealReverse;
#endif
#endif
#elif defined(CODI_FORWARD_TYPE)  // forward mode AD
#include "codi.hpp"
using mlpdouble = codi::RealForward;

#else  // primal / direct / no AD
using mlpdouble = double;
#endif
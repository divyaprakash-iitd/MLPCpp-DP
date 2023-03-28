#if defined(CODI_REVERSE_TYPE)  // reverse mode AD
#include "codi.hpp"
#include "codi/tools/data/externalFunctionUserData.hpp"

#if defined(HAVE_OMP)
using su2double = codi::RealReverseIndexOpenMP;
#else
#if defined(CODI_INDEX_TAPE)
using su2double = codi::RealReverseIndex;
#else
using su2double = codi::RealReverse;
#endif
#endif
#elif defined(CODI_FORWARD_TYPE)  // forward mode AD
#include "codi.hpp"
using su2double = codi::RealForward;

#else  // primal / direct / no AD
using su2double = double;
#endif
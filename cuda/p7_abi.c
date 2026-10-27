#include <stdio.h>

#define P7_ABI_VERSION_STR "p7_attach_abi_v1"

#ifdef __cplusplus
extern "C" {
#endif

const char* p7_abi_version(void) {
    return P7_ABI_VERSION_STR;
}

#ifdef __cplusplus
}
#endif

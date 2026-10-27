#ifndef P7_SAFETENSORS_C_H
#define P7_SAFETENSORS_C_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[256];
    char dtype[16];
    int64_t shape[4];
    int n_dims;
    uint64_t begin;
    uint64_t end;
    const void *data;
} TensorEntry;

typedef struct {
    void *mmap_ptr;
    size_t file_size;
    uint64_t header_len;
    TensorEntry *entries;
    int n_entries;
    int capacity;
} SafeTensors;

static inline const char* _st_skip_ws(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static inline const char* _st_parse_string(const char* p, char* out, size_t out_sz) {
    p = _st_skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    size_t i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p++;
        if (i + 1 < out_sz) out[i++] = *p;
        p++;
    }
    if (*p != '"') return NULL;
    out[i] = '\0';
    return p + 1;
}

static inline int safetensors_open(SafeTensors *st, const char *path) {
    memset(st, 0, sizeof(*st));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat sb;
    if (fstat(fd, &sb) < 0) { close(fd); return -1; }
    st->file_size = (size_t)sb.st_size;
    st->mmap_ptr = mmap(NULL, st->file_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (st->mmap_ptr == MAP_FAILED) return -2;

    st->header_len = *(const uint64_t*)st->mmap_ptr;
    const char *header_json = (const char*)st->mmap_ptr + 8;
    const uint8_t *data_base = (const uint8_t*)st->mmap_ptr + 8 + st->header_len;

    st->capacity = 2048;
    st->entries = (TensorEntry*)calloc(st->capacity, sizeof(TensorEntry));
    st->n_entries = 0;

    const char *p = _st_skip_ws(header_json);
    if (*p != '{') return -3;
    p++;

    while (*p && *p != '}') {
        p = _st_skip_ws(p);
        if (*p == '}') break;
        char key[256];
        p = _st_parse_string(p, key, sizeof(key));
        if (!p) break;
        p = _st_skip_ws(p);
        if (*p != ':') break;
        p++;
        p = _st_skip_ws(p);

        if (strcmp(key, "__metadata__") == 0) {
            int depth = 0;
            while (*p) {
                if (*p == '{') depth++;
                else if (*p == '}') { depth--; if (depth == 0) { p++; break; } }
                p++;
            }
        } else {
            if (*p != '{') break;
            p++;
            TensorEntry e;
            memset(&e, 0, sizeof(e));
            strncpy(e.name, key, sizeof(e.name) - 1);

            while (*p && *p != '}') {
                p = _st_skip_ws(p);
                if (*p == '}') break;
                char field[64];
                p = _st_parse_string(p, field, sizeof(field));
                if (!p) break;
                p = _st_skip_ws(p);
                if (*p != ':') break;
                p++;
                p = _st_skip_ws(p);

                if (strcmp(field, "dtype") == 0) {
                    p = _st_parse_string(p, e.dtype, sizeof(e.dtype));
                } else if (strcmp(field, "shape") == 0) {
                    if (*p == '[') {
                        p++;
                        while (*p && *p != ']') {
                            p = _st_skip_ws(p);
                            if (*p == ']') break;
                            int64_t v = 0;
                            while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
                            if (e.n_dims < 4) e.shape[e.n_dims++] = v;
                            p = _st_skip_ws(p);
                            if (*p == ',') p++;
                        }
                        if (*p == ']') p++;
                    }
                } else if (strcmp(field, "data_offsets") == 0) {
                    if (*p == '[') {
                        p++;
                        p = _st_skip_ws(p);
                        uint64_t b = 0, en = 0;
                        while (*p >= '0' && *p <= '9') { b = b * 10 + (*p - '0'); p++; }
                        p = _st_skip_ws(p);
                        if (*p == ',') p++;
                        p = _st_skip_ws(p);
                        while (*p >= '0' && *p <= '9') { en = en * 10 + (*p - '0'); p++; }
                        p = _st_skip_ws(p);
                        if (*p == ']') p++;
                        e.begin = b;
                        e.end = en;
                    }
                } else {
                    int depth = 0;
                    while (*p) {
                        if (*p == '{' || *p == '[') depth++;
                        else if (*p == '}' || *p == ']') { if (depth == 0) break; depth--; }
                        else if (*p == ',' && depth == 0) break;
                        p++;
                    }
                }
                p = _st_skip_ws(p);
                if (*p == ',') p++;
            }
            if (*p == '}') p++;
            e.data = data_base + e.begin;
            st->entries[st->n_entries++] = e;
        }
        p = _st_skip_ws(p);
        if (*p == ',') p++;
    }
    return 0;
}

static inline const TensorEntry* safetensors_find(const SafeTensors *st, const char *name) {
    for (int i = 0; i < st->n_entries; i++) {
        if (strcmp(st->entries[i].name, name) == 0) return &st->entries[i];
    }
    return NULL;
}

static inline void safetensors_close(SafeTensors *st) {
    if (st->entries) free(st->entries);
    if (st->mmap_ptr && st->mmap_ptr != MAP_FAILED) munmap(st->mmap_ptr, st->file_size);
    memset(st, 0, sizeof(*st));
}

#ifdef __cplusplus
}
#endif

#endif /* P7_SAFETENSORS_C_H */

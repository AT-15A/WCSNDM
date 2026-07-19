#ifndef WAVFILE_H
#define WAVFILE_H

#ifdef __cplusplus
extern "C" {
#endif

	// 指定浮点类型（你应定义 FP_TYPE=float）
#ifndef FP_TYPE
#define FP_TYPE float
#endif

	FP_TYPE* wavread(const char* filename, int* fs, int* nbit, int* nsmp);
	void wavwrite(const char* filename, const FP_TYPE* x, int nsmp, int fs, int nbit);

#ifdef __cplusplus
}
#endif

#endif // WAVFILE_H


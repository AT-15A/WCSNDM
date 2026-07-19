/* l1_prefix.h —— 强制包含(/FI)以给 LLSM1 全部导出符号加 l1ns_ 前缀，
   与 LLSM2 同名符号(llsm_create_frame/llsm_delete_output/llsm_harmonic_minphase 等)隔离。
   仅用于编译 LLSM1 的 .c 和 l1_path.c。 */
#ifndef L1_PREFIX_H
#define L1_PREFIX_H
#define llsm_chebyfilt            l1ns_llsm_chebyfilt
#define llsm_copy_frame           l1ns_llsm_copy_frame
#define llsm_copy_nosframe        l1ns_llsm_copy_nosframe
#define llsm_copy_sinframe        l1ns_llsm_copy_sinframe
#define llsm_create_empty_layer0  l1ns_llsm_create_empty_layer0
#define llsm_create_frame         l1ns_llsm_create_frame
#define llsm_deinit               l1ns_llsm_deinit
#define llsm_delete_frame         l1ns_llsm_delete_frame
#define llsm_delete_layer0        l1ns_llsm_delete_layer0
#define llsm_delete_layer1        l1ns_llsm_delete_layer1
#define llsm_delete_output        l1ns_llsm_delete_output
#define llsm_geometric_envelope   l1ns_llsm_geometric_envelope
#define llsm_get_iir_filter       l1ns_llsm_get_iir_filter
#define llsm_harmonic_cheaptrick  l1ns_llsm_harmonic_cheaptrick
#define llsm_harmonic_minphase    l1ns_llsm_harmonic_minphase
#define llsm_init                 l1ns_llsm_init
#define llsm_layer0_analyze       l1ns_llsm_layer0_analyze
#define llsm_layer0_phaseshift    l1ns_llsm_layer0_phaseshift
#define llsm_layer0_synthesize    l1ns_llsm_layer0_synthesize
#define llsm_layer1_from_layer0   l1ns_llsm_layer1_from_layer0
#define llsm_liprad               l1ns_llsm_liprad
#define llsm_nonuniform_envelope  l1ns_llsm_nonuniform_envelope
#define llsm_reduce_spectrum_depth l1ns_llsm_reduce_spectrum_depth
#define llsm_spectrum_from_envelope l1ns_llsm_spectrum_from_envelope
#define llsm_true_envelope        l1ns_llsm_true_envelope
#define llsm_uniform_faxis        l1ns_llsm_uniform_faxis
#define llsm_warp_freq            l1ns_llsm_warp_freq
#define spectrogram_analyze       l1ns_spectrogram_analyze
#endif

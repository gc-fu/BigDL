/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class IndexBinaryIVF extends IndexBinary {
  private transient long swigCPtr;

  protected IndexBinaryIVF(long cPtr, boolean cMemoryOwn) {
    super(swigfaissJNI.IndexBinaryIVF_SWIGUpcast(cPtr), cMemoryOwn);
    swigCPtr = cPtr;
  }

  protected static long getCPtr(IndexBinaryIVF obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_IndexBinaryIVF(swigCPtr);
      }
      swigCPtr = 0;
    }
    super.delete();
  }

  public void setInvlists(InvertedLists value) {
    swigfaissJNI.IndexBinaryIVF_invlists_set(swigCPtr, this, InvertedLists.getCPtr(value), value);
  }

  public InvertedLists getInvlists() {
    long cPtr = swigfaissJNI.IndexBinaryIVF_invlists_get(swigCPtr, this);
    return (cPtr == 0) ? null : new InvertedLists(cPtr, false);
  }

  public void setOwn_invlists(boolean value) {
    swigfaissJNI.IndexBinaryIVF_own_invlists_set(swigCPtr, this, value);
  }

  public boolean getOwn_invlists() {
    return swigfaissJNI.IndexBinaryIVF_own_invlists_get(swigCPtr, this);
  }

  public void setNprobe(long value) {
    swigfaissJNI.IndexBinaryIVF_nprobe_set(swigCPtr, this, value);
  }

  public long getNprobe() {
    return swigfaissJNI.IndexBinaryIVF_nprobe_get(swigCPtr, this);
  }

  public void setMax_codes(long value) {
    swigfaissJNI.IndexBinaryIVF_max_codes_set(swigCPtr, this, value);
  }

  public long getMax_codes() {
    return swigfaissJNI.IndexBinaryIVF_max_codes_get(swigCPtr, this);
  }

  public void setUse_heap(boolean value) {
    swigfaissJNI.IndexBinaryIVF_use_heap_set(swigCPtr, this, value);
  }

  public boolean getUse_heap() {
    return swigfaissJNI.IndexBinaryIVF_use_heap_get(swigCPtr, this);
  }

  public void setDirect_map(DirectMap value) {
    swigfaissJNI.IndexBinaryIVF_direct_map_set(swigCPtr, this, DirectMap.getCPtr(value), value);
  }

  public DirectMap getDirect_map() {
    long cPtr = swigfaissJNI.IndexBinaryIVF_direct_map_get(swigCPtr, this);
    return (cPtr == 0) ? null : new DirectMap(cPtr, false);
  }

  public void setQuantizer(IndexBinary value) {
    swigfaissJNI.IndexBinaryIVF_quantizer_set(swigCPtr, this, getCPtr(value), value);
  }

  public IndexBinary getQuantizer() {
    long cPtr = swigfaissJNI.IndexBinaryIVF_quantizer_get(swigCPtr, this);
    return (cPtr == 0) ? null : new IndexBinary(cPtr, false);
  }

  public void setNlist(long value) {
    swigfaissJNI.IndexBinaryIVF_nlist_set(swigCPtr, this, value);
  }

  public long getNlist() {
    return swigfaissJNI.IndexBinaryIVF_nlist_get(swigCPtr, this);
  }

  public void setOwn_fields(boolean value) {
    swigfaissJNI.IndexBinaryIVF_own_fields_set(swigCPtr, this, value);
  }

  public boolean getOwn_fields() {
    return swigfaissJNI.IndexBinaryIVF_own_fields_get(swigCPtr, this);
  }

  public void setCp(ClusteringParameters value) {
    swigfaissJNI.IndexBinaryIVF_cp_set(swigCPtr, this, ClusteringParameters.getCPtr(value), value);
  }

  public ClusteringParameters getCp() {
    long cPtr = swigfaissJNI.IndexBinaryIVF_cp_get(swigCPtr, this);
    return (cPtr == 0) ? null : new ClusteringParameters(cPtr, false);
  }

  public void setClustering_index(Index value) {
    swigfaissJNI.IndexBinaryIVF_clustering_index_set(swigCPtr, this, Index.getCPtr(value), value);
  }

  public Index getClustering_index() {
    long cPtr = swigfaissJNI.IndexBinaryIVF_clustering_index_get(swigCPtr, this);
    return (cPtr == 0) ? null : new Index(cPtr, false);
  }

  public IndexBinaryIVF(IndexBinary quantizer, long d, long nlist) {
    this(swigfaissJNI.new_IndexBinaryIVF__SWIG_0(getCPtr(quantizer), quantizer, d, nlist), true);
  }

  public IndexBinaryIVF() {
    this(swigfaissJNI.new_IndexBinaryIVF__SWIG_1(), true);
  }

  public void reset() {
    swigfaissJNI.IndexBinaryIVF_reset(swigCPtr, this);
  }

  public void train(int n, SWIGTYPE_p_unsigned_char x) {
    swigfaissJNI.IndexBinaryIVF_train(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x));
  }

  public void add(int n, SWIGTYPE_p_unsigned_char x) {
    swigfaissJNI.IndexBinaryIVF_add(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x));
  }

  public void add_with_ids(int n, SWIGTYPE_p_unsigned_char x, SWIGTYPE_p_long xids) {
    swigfaissJNI.IndexBinaryIVF_add_with_ids(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), SWIGTYPE_p_long.getCPtr(xids));
  }

  public void add_core(int n, SWIGTYPE_p_unsigned_char x, SWIGTYPE_p_long xids, SWIGTYPE_p_long precomputed_idx) {
    swigfaissJNI.IndexBinaryIVF_add_core(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), SWIGTYPE_p_long.getCPtr(xids), SWIGTYPE_p_long.getCPtr(precomputed_idx));
  }

  public void search_preassigned(int n, SWIGTYPE_p_unsigned_char x, int k, SWIGTYPE_p_long assign, SWIGTYPE_p_int centroid_dis, SWIGTYPE_p_int distances, SWIGTYPE_p_long labels, boolean store_pairs, IVFSearchParameters params) {
    swigfaissJNI.IndexBinaryIVF_search_preassigned__SWIG_0(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), k, SWIGTYPE_p_long.getCPtr(assign), SWIGTYPE_p_int.getCPtr(centroid_dis), SWIGTYPE_p_int.getCPtr(distances), SWIGTYPE_p_long.getCPtr(labels), store_pairs, IVFSearchParameters.getCPtr(params), params);
  }

  public void search_preassigned(int n, SWIGTYPE_p_unsigned_char x, int k, SWIGTYPE_p_long assign, SWIGTYPE_p_int centroid_dis, SWIGTYPE_p_int distances, SWIGTYPE_p_long labels, boolean store_pairs) {
    swigfaissJNI.IndexBinaryIVF_search_preassigned__SWIG_1(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), k, SWIGTYPE_p_long.getCPtr(assign), SWIGTYPE_p_int.getCPtr(centroid_dis), SWIGTYPE_p_int.getCPtr(distances), SWIGTYPE_p_long.getCPtr(labels), store_pairs);
  }

  public SWIGTYPE_p_faiss__BinaryInvertedListScanner get_InvertedListScanner(boolean store_pairs) {
    long cPtr = swigfaissJNI.IndexBinaryIVF_get_InvertedListScanner__SWIG_0(swigCPtr, this, store_pairs);
    return (cPtr == 0) ? null : new SWIGTYPE_p_faiss__BinaryInvertedListScanner(cPtr, false);
  }

  public SWIGTYPE_p_faiss__BinaryInvertedListScanner get_InvertedListScanner() {
    long cPtr = swigfaissJNI.IndexBinaryIVF_get_InvertedListScanner__SWIG_1(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_faiss__BinaryInvertedListScanner(cPtr, false);
  }

  public void search(int n, SWIGTYPE_p_unsigned_char x, int k, SWIGTYPE_p_int distances, SWIGTYPE_p_long labels) {
    swigfaissJNI.IndexBinaryIVF_search(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), k, SWIGTYPE_p_int.getCPtr(distances), SWIGTYPE_p_long.getCPtr(labels));
  }

  public void range_search(int n, SWIGTYPE_p_unsigned_char x, int radius, RangeSearchResult result) {
    swigfaissJNI.IndexBinaryIVF_range_search(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), radius, RangeSearchResult.getCPtr(result), result);
  }

  public void reconstruct(int key, SWIGTYPE_p_unsigned_char recons) {
    swigfaissJNI.IndexBinaryIVF_reconstruct(swigCPtr, this, key, SWIGTYPE_p_unsigned_char.getCPtr(recons));
  }

  public void reconstruct_n(int i0, int ni, SWIGTYPE_p_unsigned_char recons) {
    swigfaissJNI.IndexBinaryIVF_reconstruct_n(swigCPtr, this, i0, ni, SWIGTYPE_p_unsigned_char.getCPtr(recons));
  }

  public void search_and_reconstruct(int n, SWIGTYPE_p_unsigned_char x, int k, SWIGTYPE_p_int distances, SWIGTYPE_p_long labels, SWIGTYPE_p_unsigned_char recons) {
    swigfaissJNI.IndexBinaryIVF_search_and_reconstruct(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), k, SWIGTYPE_p_int.getCPtr(distances), SWIGTYPE_p_long.getCPtr(labels), SWIGTYPE_p_unsigned_char.getCPtr(recons));
  }

  public void reconstruct_from_offset(int list_no, int offset, SWIGTYPE_p_unsigned_char recons) {
    swigfaissJNI.IndexBinaryIVF_reconstruct_from_offset(swigCPtr, this, list_no, offset, SWIGTYPE_p_unsigned_char.getCPtr(recons));
  }

  public long remove_ids(IDSelector sel) {
    return swigfaissJNI.IndexBinaryIVF_remove_ids(swigCPtr, this, IDSelector.getCPtr(sel), sel);
  }

  public void merge_from(IndexBinaryIVF other, int add_id) {
    swigfaissJNI.IndexBinaryIVF_merge_from(swigCPtr, this, IndexBinaryIVF.getCPtr(other), other, add_id);
  }

  public long get_list_size(long list_no) {
    return swigfaissJNI.IndexBinaryIVF_get_list_size(swigCPtr, this, list_no);
  }

  public void make_direct_map(boolean new_maintain_direct_map) {
    swigfaissJNI.IndexBinaryIVF_make_direct_map__SWIG_0(swigCPtr, this, new_maintain_direct_map);
  }

  public void make_direct_map() {
    swigfaissJNI.IndexBinaryIVF_make_direct_map__SWIG_1(swigCPtr, this);
  }

  public void set_direct_map_type(DirectMap.Type type) {
    swigfaissJNI.IndexBinaryIVF_set_direct_map_type(swigCPtr, this, type.swigValue());
  }

  public void replace_invlists(InvertedLists il, boolean own) {
    swigfaissJNI.IndexBinaryIVF_replace_invlists__SWIG_0(swigCPtr, this, InvertedLists.getCPtr(il), il, own);
  }

  public void replace_invlists(InvertedLists il) {
    swigfaissJNI.IndexBinaryIVF_replace_invlists__SWIG_1(swigCPtr, this, InvertedLists.getCPtr(il), il);
  }

}